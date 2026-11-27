"""로깅 유틸 — 콘솔 / TensorBoard / CSV 를 한 곳에서 처리.

알고리즘 코드가 로깅 세부사항(파일 경로, 텐서보드 유무 등)을 신경쓰지
않도록 분리했다. 알고리즘은 `logger.log_train(...)` 처럼 부르기만 하면 된다.
"""
import csv
import os

from utils.collision import classify


class RunLogger:
    """한 번의 학습 실행(run)에 대한 로그를 모아 기록한다.

    - 콘솔: 사람이 읽을 한 줄 요약
    - TensorBoard: results/<run>/tb/ 에 스칼라 기록 (tensorboard 없으면 자동 생략)
    - CSV: training_log.csv(학습 곡선), eval_log.csv(주행 지표)
           → close() 또는 save_csv() 호출 시 파일로 씀
    """

    def __init__(self, run_dir: str, use_tensorboard: bool = True,
                 resume_step: int | None = None):
        """로거를 만든다.

        Args:
            run_dir: 학습 결과 폴더.
            use_tensorboard: TensorBoard 기록 여부.
            resume_step: 이어 학습을 시작하는 스텝. 주어지면 기존 CSV에서 이 스텝
                이하의 기록만 이어받고, TensorBoard에서는 이 스텝 이후 기록을 지운다.
        """
        self.run_dir = run_dir
        self.train_rows = []   # 학습 곡선 기록
        self.eval_rows = []    # 평가 지표 기록
        self.writer = None
        if resume_step is not None:
            self.train_rows = self._read_rows("training_log.csv", resume_step)
            self.eval_rows = self._read_rows("eval_log.csv", resume_step)

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                purge_step = None if resume_step is None else resume_step + 1
                self.writer = SummaryWriter(os.path.join(run_dir, "tb"),
                                            purge_step=purge_step)
            except Exception as e:
                # tensorboard 미설치, 또는 경로 문제(한글 경로 등)로 실패해도
                # 학습 자체는 계속되어야 하므로 CSV만 쓰도록 넘어간다.
                print(f"[안내] TensorBoard 로깅 비활성화 ({type(e).__name__}). "
                      f"CSV 로그만 기록합니다.")
                self.writer = None

    # ------------------------------------------------------------------
    def log_train(self, step: int, metrics: dict, verbose: bool = True):
        """업데이트 1회의 학습 지표 기록."""
        row = {"steps": step, **metrics}
        self.train_rows.append(row)

        if self.writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"train/{k}", v, step)

        if verbose:
            parts = [f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                     for k, v in metrics.items() if k != "update"]
            upd = metrics.get("update", "")
            print(f"[update {upd:>4}] steps={step:>7d}  " + "  ".join(parts))

    def log_eval(self, step: int, metrics: dict, summary: str = "",
                 n_episodes: int = 0):
        """주기적 평가 결과 기록."""
        self.eval_rows.append({"steps": step, **metrics})

        if self.writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"eval/{k}", v, step)

        print(f"    [eval @{step}] {n_episodes}ep  {summary}")

    def log_collision(self, step: int, phase: str, demand_scale: float, record: dict):
        """충돌 기록 1건을 ``run_dir/collisions.csv``에 바로 덧붙인다.

        Args:
            step: 충돌이 난 시점의 누적 학습 스텝.
            phase: ``train``이면 학습 rollout, ``eval``이면 학습 중 평가다.
            demand_scale: 충돌 당시 교통량 배율.
            record: :meth:`~env.sumo_env.SumoHighwayEnv.step`의 ``info["collision"]``.
        """
        row = {"steps": step, "phase": phase, "demand_scale": demand_scale,
               "category": classify(record), **record}
        path = os.path.join(self.run_dir, "collisions.csv")
        write_header = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def log_episode(self, step: int, ep_return: float, ep_length: int):
        """학습 중 개별 에피소드가 끝날 때 (TensorBoard 전용, 콘솔 출력 없음)."""
        if self.writer is not None:
            self.writer.add_scalar("episode/return", ep_return, step)
            self.writer.add_scalar("episode/length", ep_length, step)

    def _read_rows(self, name: str, max_step: int) -> list[dict]:
        """기존 CSV에서 ``steps``가 ``max_step`` 이하인 행을 읽는다. 파일이 없으면 빈 목록이다.

        Args:
            name: ``run_dir`` 안의 CSV 파일 이름.
            max_step: 남길 마지막 스텝.
        """
        path = os.path.join(self.run_dir, name)
        if not os.path.exists(path):
            return []
        with open(path, newline="", encoding="utf-8") as f:
            return [row for row in csv.DictReader(f) if int(row["steps"]) <= max_step]

    # ------------------------------------------------------------------
    def save_csv(self):
        """모아둔 기록을 CSV 파일로 저장."""
        for name, rows in [("training_log.csv", self.train_rows),
                           ("eval_log.csv", self.eval_rows)]:
            if not rows:
                continue
            # 모든 행의 키를 합집합으로 모아 헤더를 만든다 (키가 달라도 안전)
            fields = list(dict.fromkeys(k for r in rows for k in r))
            with open(os.path.join(self.run_dir, name), "w", newline="",
                      encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

    def close(self):
        self.save_csv()
        if self.writer is not None:
            self.writer.close()
