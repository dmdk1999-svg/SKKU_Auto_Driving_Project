"""정책 평가 유틸 — 주행 지표 측정.

알고리즘과 분리해둔 이유: 평가는 "정책이 무엇으로 학습됐는지"와 무관하다.
`predict(obs, deterministic=True) -> action` 메서드만 있으면
PPO든 SAC든, 심지어 규칙 기반 컨트롤러든 똑같이 평가할 수 있다.

왜 별도 평가가 필요한가:
    학습 중 에피소드는 탐험 노이즈(σ)가 섞인 확률적 행동의 결과라
    "지금 정책의 실제 운전 실력"을 정확히 보여주지 못한다.
    그래서 주기적으로 노이즈 없는(deterministic) 정책으로 주행시켜
    충돌률 / 완주율 / 평균속도 / 차간거리를 따로 측정한다.
"""
from collections.abc import Callable
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class DrivingMetrics:
    """평가 1회(n_episodes 평균)의 주행 지표."""
    collision_rate: float   # 충돌로 끝난 에피소드 비율 (0~1, 낮을수록 좋음)
    red_light_rate: float   # 적색 신호 통과로 끝난 비율 (0~1, 낮을수록 좋음)
    success_rate: float     # 도로 끝까지 완주한 비율 (0~1, 높을수록 좋음)
    exit_miss_rate: float   # 출구를 놓쳐 끝난 비율 (0~1, 낮을수록 좋음)
    timeout_rate: float     # 시간초과로 끝난 비율 (= 1 - 충돌 - 신호위반 - 출구 놓침 - 완주)
    mean_speed: float       # 전 스텝 평균 주행속도 (m/s)
    mean_gap: float         # 선행차 감지 스텝에서의 평균 차간거리 (m)
    min_gap: float          # 관측된 최소 차간거리 (m) — 위험 운전 지표
    ep_return: float        # 에피소드당 평균 누적 보상
    ep_length: float        # 에피소드당 평균 스텝 수
    lane_changes: float     # 에피소드당 평균 "실행된" 차선변경 횟수
                            # (추월 시나리오에서 정책이 실제로 차선을
                            #  활용하는지 보는 핵심 지표)

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        """콘솔 한 줄 출력용 문자열."""
        return (f"충돌률={self.collision_rate:.0%}  "
                f"신호위반={self.red_light_rate:.0%}  "
                f"출구놓침={self.exit_miss_rate:.0%}  "
                f"완주율={self.success_rate:.0%}  "
                f"평균속도={self.mean_speed:.1f}m/s  "
                f"평균차간={self.mean_gap:.1f}m  "
                f"최소차간={self.min_gap:.1f}m  "
                f"차선변경={self.lane_changes:.1f}회  "
                f"보상={self.ep_return:.1f}")


@dataclass
class _Episode:
    """에피소드 하나의 진행 중 집계."""
    ret: float = 0.0
    length: int = 0
    lane_changes: int = 0
    ended_by: str | None = None

    def add(self, reward: float, info: dict, speeds: list[float], gaps: list[float],
            on_collision: Callable[[dict], None] | None) -> None:
        """한 스텝의 보상과 ``info``를 반영하고, 속도와 차간거리를 ``speeds``, ``gaps``에 덧붙인다.

        Args:
            reward: 이번 스텝 보상.
            info: ``env.step``이 돌려준 info.
            speeds: 전체 평가의 속도 표본.
            gaps: 전체 평가의 차간거리 표본.
            on_collision: 주어지면 충돌마다 ``info["collision"]`` 기록으로 호출한다.
        """
        self.ret += reward
        self.length += 1

        # 실행된 차선변경 집계 (유지 명령의 changeLane 핀 동작은 제외)
        if info.get("lane_change", 0) != 0 and info.get("lane_change_applied"):
            self.lane_changes += 1

        if on_collision is not None and info.get("collision") is not None:
            on_collision(info["collision"])
        if info.get("collided"):
            self.ended_by = "collision"
        elif info.get("red_light"):
            self.ended_by = "red_light"
        elif info.get("exit_missed"):
            self.ended_by = "exit_miss"
        elif info.get("arrived"):
            self.ended_by = "arrival"

        # 속도/차간거리는 env가 info로 넘겨주는 원시값(m/s, m)을 쓴다.
        # 관측(obs) 레이아웃이 바뀌어도 평가 코드는 영향받지 않는다.
        # 종료 스텝은 None (ego가 이미 제거됨) → 통계에서 제외.
        if info.get("speed") is not None:
            speeds.append(float(info["speed"]))
        if info.get("gap") is not None:
            gaps.append(float(info["gap"]))


def _metrics(episodes: list[_Episode], speeds: list[float], gaps: list[float],
             visibility: float) -> DrivingMetrics:
    """끝난 에피소드들의 집계로 :class:`DrivingMetrics`를 만든다.

    Args:
        episodes: 끝난 에피소드들.
        speeds: 전체 평가의 속도 표본.
        gaps: 전체 평가의 차간거리 표본. 비어 있으면 차간거리 지표는 ``visibility``다.
        visibility: 선행차 감지 거리(m).
    """
    n = len(episodes)
    ended = [ep.ended_by for ep in episodes]
    return DrivingMetrics(
        collision_rate=ended.count("collision") / n,
        red_light_rate=ended.count("red_light") / n,
        exit_miss_rate=ended.count("exit_miss") / n,
        success_rate=ended.count("arrival") / n,
        timeout_rate=ended.count(None) / n,
        mean_speed=float(np.mean(speeds)) if speeds else 0.0,
        mean_gap=float(np.mean(gaps)) if gaps else float(visibility),
        min_gap=float(np.min(gaps)) if gaps else float(visibility),
        ep_return=float(np.mean([ep.ret for ep in episodes])),
        ep_length=float(np.mean([ep.length for ep in episodes])),
        lane_changes=float(np.mean([ep.lane_changes for ep in episodes])),
    )


def evaluate_policy(policy, env, n_episodes: int = 5,
                    deterministic: bool = True,
                    on_collision: Callable[[dict], None] | None = None,
                    verbose: bool = False) -> DrivingMetrics:
    """정책을 n_episodes 주행시키고 주행 지표 평균을 반환.

    Args:
        policy: `predict(obs, deterministic) -> action` 을 가진 객체
                (PPOAgent, 또는 같은 인터페이스의 무엇이든)
        env:    SumoHighwayEnv (mean_gap 기본값용 visibility 속성만 사용)
        n_episodes: 평가 에피소드 수. 많을수록 지표가 안정적이지만 느리다
        deterministic: True면 탐험 노이즈 없이 평균 행동만 사용
        on_collision: 주어지면 충돌마다 ``info["collision"]`` 기록으로 호출한다.
        verbose: True면 에피소드가 끝날 때마다 결과를 한 줄 출력

    Note:
        모든 지표는 env.step()의 info(collided/arrived/speed/gap)에서 읽는다.
        관측(obs) 벡터의 구성과 무관하므로, 관측을 바꿔도 이 코드는 안 깨진다.
    """
    episodes, speeds, gaps = [], [], []
    for ep_idx in range(n_episodes):
        obs, _ = env.reset()
        ep, done = _Episode(), False
        while not done:
            action = policy.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep.add(reward, info, speeds, gaps, on_collision)
            done = terminated or truncated
        episodes.append(ep)
        if verbose:
            print(f"  [{ep_idx + 1}/{n_episodes}] {ep.ended_by or 'timeout'}  "
                  f"보상={ep.ret:.1f}  길이={ep.length}  차선변경={ep.lane_changes}",
                  flush=True)
    return _metrics(episodes, speeds, gaps, env.visibility)


def evaluate_policy_async(policy, envs, start_obs: list, total_steps: int,
                          n_episodes: int, visibility: float,
                          deterministic: bool = True,
                          on_collision: Callable[[dict], None] | None = None
                          ) -> DrivingMetrics:
    """:class:`~env.vec_env.SubprocEnvs`의 환경들에서 평가 에피소드를 나눠 돌리고 지표를 반환한다.

    모든 환경은 응답 대기 중인 명령이 없어야 한다. 환경마다 행동을 받을 수 있게 되는 대로
    다음 행동을 보내고, 에피소드가 끝난 환경은 남은 에피소드가 있으면 바로 리셋한다.
    에피소드를 모두 ``n_episodes``개 시작하고, 시작한 에피소드가 모두 끝나면 반환한다.
    반환 시점에 모든 환경은 에피소드가 끝났거나 아무 에피소드도 시작하지 않은 상태다.

    Args:
        policy: ``predict_batch(obs, deterministic) -> actions``를 가진 객체.
        envs: 평가할 환경 묶음.
        start_obs: 환경별로 막 리셋해 얻은 초기 관측이다. ``None``인 환경은 리셋부터 한다.
        total_steps: 리셋할 때 환경에 넘길 전체 학습 스텝 수.
        n_episodes: 평가 에피소드 수.
        visibility: 선행차 감지 거리(m).
        deterministic: True면 탐험 노이즈 없이 평균 행동만 쓴다.
        on_collision: 주어지면 충돌마다 ``info["collision"]`` 기록으로 호출한다.
    """
    obs = list(start_obs)
    running: dict[int, _Episode] = {}
    waiting: set[int] = set()
    resetting: set[int] = set()
    episodes, speeds, gaps = [], [], []
    launched = 0

    def start(i: int) -> None:
        nonlocal launched
        launched += 1
        running[i] = _Episode()
        if obs[i] is None:
            envs.send(i, "reset", total_steps)
            waiting.add(i)
            resetting.add(i)

    for i in range(min(envs.n_envs, n_episodes)):
        start(i)
    while running:
        ready = [i for i in running if i not in waiting]
        if ready:
            actions = policy.predict_batch(np.stack([obs[i] for i in ready]),
                                           deterministic=deterministic)
            for k, i in enumerate(ready):
                envs.send(i, "step", actions[k])
                waiting.add(i)
        for i in envs.wait(sorted(waiting)):
            result = envs.recv(i)
            waiting.discard(i)
            if i in resetting:
                resetting.discard(i)
                obs[i] = result
                continue
            next_obs, reward, terminated, truncated, info = result
            running[i].add(reward, info, speeds, gaps, on_collision)
            if terminated or truncated:
                episodes.append(running.pop(i))
                obs[i] = None
                if launched < n_episodes:
                    start(i)
            else:
                obs[i] = next_obs
    return _metrics(episodes, speeds, gaps, visibility)
