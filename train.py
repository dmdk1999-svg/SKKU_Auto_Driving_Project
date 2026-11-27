"""학습 실행 스크립트.

╔══════════════════════════════════════════════════════════════════╗
║  실험할 때 만지는 곳은 아래 "하이퍼파라미터" 블록이다.              ║
║  gamma, lr, n_steps 등 학습 설정을 여기서 바로 바꾸면 된다.        ║
║                                                                  ║
║  그 외의 것들:                                                    ║
║    도로/교통 설정   → env/road_config.py                          ║
║    MDP(관측/보상)   → env/mdp_config.py                           ║
║    알고리즘 코드    → algorithms/ppo.py                           ║
║    네트워크 / 평가  → utils/networks.py, utils/evaluator.py       ║
╚══════════════════════════════════════════════════════════════════╝

실행:  python train.py
"""
import os
import shutil
from datetime import datetime

from env.road_config import ROAD, EGO, TRAFFIC
from env.mdp_config import SIMULATION, OBSERVATION, ACTION
from env.mdp_config import REWARD as REWARD_DEFAULTS
from env import road_builder
from env.sumo_env import SumoHighwayEnv

from algorithms.ppo import PPO
from utils.buffer import RolloutBuffer
from utils.logger import RunLogger

BASE = os.path.dirname(os.path.abspath(__file__))


# ══════════════════════════════════════════════════════════════════
#  하이퍼파라미터 — 실험은 여기서
# ══════════════════════════════════════════════════════════════════
TOTAL_TIMESTEPS = 5_000_000   # 총 학습 스텝 수

# ══════════════════════════════════════════════════════════════════
#  보상 튜닝 — env/mdp_config.py 의 기본값을 여기서 덮어쓴다.
#  (여기 적은 키만 바뀌고, 나머지는 mdp_config 기본값 그대로)
#  test.py 도 이 REWARD 를 import 하므로 학습/평가 조건이 항상 일치한다.
#  이 파일이 run 폴더에 스냅샷되므로 "그 결과가 어떤 계수였는지"도 남는다.
# ══════════════════════════════════════════════════════════════════
REWARD_OVERRIDES = dict(
    speed_weight=0.1,          # 진행 거리 보상 (스텝당 최대 이 값)
    collision_penalty=10.0,    # 충돌 시 -이 값, 즉시 종료
    red_light_penalty=9.0,     # 적색 신호 통과 시 -이 값, 즉시 종료
    arrival_bonus=2.0,         # 완주 보너스
    safe_gap_penalty=0.1,      # UN R157 최소 차간거리 위반 시 스텝당 최대 감점
    slow_penalty=0.1,          # 속도에 비해 앞 장애물과의 간격이 넓으면 스텝당 최대 감점
    slow_tolerance=0.2,        #   제한속도보다 20%까지 느린 건 허용
    yellow_pass_penalty=3.0,   # 처음 봤을 때 정지할 수 있었던 황색 신호 통과 시 1회 감점
    exit_miss_penalty=6.0,     # 출구 놓침(경로로 이어지지 않는 차선 끝 도달) 시 감점 후 종료
    queue_cut_penalty=0.3,     # 합류 완료 시 밀려난 대기 줄 차량 1대당 감점
    queue_cut_max_penalty=3.0, #   합류 1회당 대기 줄 앞지르기 감점 상한 (0=상한 없음)
    overlap_penalty=0.05,      # 차선 경계에 걸친 폭에 비례한 스텝당 감점
    overlap_grace=3.0,         #   처음 걸친 뒤 이 시간(s) 동안은 겹침 감점 없음
    junction_overlap_penalty=0.5,  # 차선에 걸친 채 신호등 교차로 진입 시 1회 감점
    blocked_penalty=0.05,      # 더 빠른 인접 차선을 두고 머무는 기회비용 감점 (스텝당 최대)
    nav_shaping_weight=0.5,    # 출구 접근 시 최적 차선 포텐셜 크기 (차선 1칸당 최대, 0=비활성)
    nav_shaping_horizon=300.0, # 내비게이션 셰이핑 시작 잔여거리 (m)
    invalid_action_penalty=0.0,# 차선이 없는 방향 차선변경 트리거 감점
)
REWARD = {**REWARD_DEFAULTS, **REWARD_OVERRIDES}

# ************************ WEEK 6 과제를 위해 건드릴 부분 *********************************
# train.py는 어디서 뭘 불러오는지 등을 정리한 파일인거임.
# step은 바꾸지 말것.
HPARAMS = dict(
    # ---- 최적화 ----
    # lr=1e-4,                 # 학습률. 보상이 요동치면 낮출 것 (3e-4 → 1e-4 → 5e-5)
    lr=1.5e-4,
    n_steps=8192,            # rollout 길이. 크면 advantage 추정이 안정적
    n_epochs=10,             # 모은 데이터 재사용 횟수
    minibatch_size=256,

    # ---- 할인 / advantage ----
    # gamma=0.99,              # 할인율 γ. 0.99 ≈ 100스텝(50초) 앞을 내다봄
    gamma=0.997,
    gae_lambda=0.95,         # GAE λ. 1에 가까울수록 분산↑편향↓

    # ---- PPO 고유 ----
    clip_eps=0.2,            # 클리핑 범위 ε
    value_coef=0.5,          # 가치 손실 가중치
    entropy_coef=0.0,
    lane_entropy_coef=0.01,  # 차선 head "만"의 엔트로피 보너스 (hybrid 전용).
                             #   차선변경 확률의 조기 붕괴(p_lane_change→0)를
                             #   막아 추월 이득을 경험할 시간을 벌어준다.
                             #   가감속 σ에는 영향 없음 (수렴 방해 안 함).
                             #   추월을 배운 뒤에는 0.005나 0으로 낮춰도 됨.        # 엔트로피 보너스.
                             #   연속 제어에서는 0이 표준. 0.01처럼 크게 주면
                             #   σ가 안 줄어들어 정책이 계속 랜덤에 머문다.
    max_grad_norm=0.5,
    normalize_advantage=True,

    # ---- 네트워크 (utils/networks.py) ----
    hidden_sizes=(256, 256),
    activation="relu",       # "tanh" | "relu" | "elu"
    policy_type="hybrid",      # "hybrid": 가감속·기동 횡가속도는 연속(Normal),
                               #   차선변경 트리거는 3-way 이산(Categorical).
                               #   평가(argmax)에서도 학습한 차선변경이 그대로 나온다.
                               #   (아래 lane_keep_bias, lane_entropy_coef는 hybrid 전용)
                               # "gaussian": 3차원 모두 연속(Normal). 트리거는 환경이
                               #   ±0.5로 양자화하므로 평가(μ)에서 변경이 잘 나오지 않는다.
    lane_keep_bias=0.5,        # hybrid 차선 head의 "유지" 초기 편향.
                               #   0.5 → 초기 (좌27%/유지45%/우27%) ≈ 55% 변경 시도 (공격적)
                               #   1.0 → ≈32% 시도 (보수적)
                               #   낮추면 탐험↑, 높이면 보수적
    init_log_std=-0.5,         # 연속 축 초기 탐험 폭 σ=e^-0.5≈0.61
                               #   hybrid: 스칼라면 가감속·횡가속도 공통,
                               #   (가감속, 횡가속도) 튜플이면 축별로 지정.
                               #   차선 트리거 탐험은 위 lane_keep_bias가 담당한다.

    # ---- 기타 ----
    seed=0,
    device="auto",           # "auto" | "cpu" | "cuda"
)

# ---- 평가 / 로깅 ----
EVAL_INTERVAL = 1       # 몇 번의 업데이트마다 주행 지표를 평가할지 (0이면 안 함)
EVAL_EPISODES = 10      # 평가 1회당 에피소드 수
LOG_INTERVAL = 1        # 몇 번의 업데이트마다 콘솔에 출력할지
RESULTS_DIR = "results" # 결과 저장 폴더
RUN_NAME = None         # None이면 run_날짜_시각 자동 생성.
                        # 문자열을 주면 그 이름으로 저장 (예: "exp_lr1e-4")
RESUME = "results/run_20261127_195432/checkpoints/ckpt_401408.pt"  # 이어 학습할 체크포인트 경로. None이면 처음부터 학습.
                        # 예: "results/run_20261126_134914/checkpoints/ckpt_102400.pt"
                        # 그 체크포인트의 run 폴더에 이어서 기록하고, 이후 스텝의 기존 로그는 버린다.
                        # TOTAL_TIMESTEPS는 이어 학습한 스텝을 포함한 전체 스텝 수다.
                        # 학습률은 체크포인트 값이 아니라 위 HPARAMS의 lr을 쓴다.
# ══════════════════════════════════════════════════════════════════


def make_run_dir() -> str:
    """results/<run이름>/ 폴더를 만들고 경로를 반환."""
    name = RUN_NAME or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(BASE, RESULTS_DIR, name)
    os.makedirs(run_dir, exist_ok=True)
    if not os.path.isdir(run_dir):
        raise NotADirectoryError(
            f"결과 폴더를 만들 수 없습니다: {run_dir}\n"
            f"  같은 이름의 '파일'이 이미 있는지 확인하세요.")
    return run_dir


def snapshot_configs(run_dir: str):
    """이 run에 사용된 설정 파일들을 복사해둔다 (재현/비교용).

    학습 스크립트 자신(train.py)도 함께 저장한다 — 하이퍼파라미터가
    이 파일 안에 있으므로, 나중에 "이 결과가 어떤 설정이었나"를
    정확히 되짚을 수 있다.
    """
    for src in [os.path.join(BASE, "env", "road_config.py"),
                os.path.join(BASE, "env", "mdp_config.py"),
                os.path.join(BASE, "train.py")]:
        shutil.copy(src, run_dir)


if __name__ == "__main__":
    # 1) 도로 생성 (env/road_config.py 기준으로 매번 재생성)
    sumocfg = road_builder.build(ROAD, EGO, TRAFFIC,
                                 os.path.join(BASE, "env", "sumo"))
    print(f"도로 생성 완료: {sumocfg}")
    if road_builder.osm_config(ROAD) is not None:
        print(f"  OSM 도로망, ego 경로 {ROAD['osm']['ego_routes']}")
    else:
        print(f"  길이 {ROAD['length']}m, {ROAD['num_lanes']}차선, "
              f"교통량 {TRAFFIC['vehs_per_hour']}대/시간")

    # 2) 결과 폴더. 이어 학습이면 체크포인트가 있는 run 폴더를 그대로 쓴다.
    if RESUME:
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(RESUME)))
    else:
        run_dir = make_run_dir()
    print(f"결과 폴더: {run_dir}")

    # 3) 환경
    env = SumoHighwayEnv(
        cfg_path=sumocfg,
        road=ROAD, ego=EGO,
        mdp_sim=SIMULATION, mdp_obs=OBSERVATION,
        action=ACTION, reward=REWARD,
        traffic=TRAFFIC,
        gui=False,   # 학습은 화면 없이 (수십 배 빠름)
        training=True,   # 배경차 비상제동 커리큘럼 사용 (road_config TRAFFIC 참고)
    )

    # 4) 경험 버퍼 + 에이전트
    #    버퍼는 메인에서 만들어 에이전트에 "주입"한다 (utils/buffer.py).
    #    PPO는 on-policy라 RolloutBuffer를 쓰고, 나중에 SAC/DQN 같은
    #    off-policy 알고리즘을 붙일 땐 여기서 ReplayBuffer로 갈아끼우면 된다.
    buffer = RolloutBuffer()
    agent = PPO(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
        buffer=buffer,
        **HPARAMS,
    )
    print(f"obs_dim={env.observation_space.shape[0]}, "
          f"act_dim={env.action_space.shape[0]}, device={agent.device}")
    if RESUME:
        extra = agent.load_checkpoint(RESUME)
        if "demand_scale" in extra:
            env.demand_scale = float(extra["demand_scale"])
        print(f"이어 학습: {RESUME} (steps={agent.num_timesteps}, "
              f"update={agent.n_updates}, best={agent.best_score:.2f})")
    logger = RunLogger(run_dir,
                       resume_step=agent.num_timesteps if RESUME else None)

    # 5) 학습 → 저장
    #    try/finally: Ctrl+C나 에러로 중단돼도 SUMO 정리 + 결과 저장은 수행
    try:
        agent.learn(env,
                    total_timesteps=TOTAL_TIMESTEPS,
                    logger=logger,
                    log_interval=LOG_INTERVAL,
                    eval_interval=EVAL_INTERVAL,
                    eval_episodes=EVAL_EPISODES,
                    run_dir=run_dir)
    finally:
        env.close()
        logger.close()
        model_path = os.path.join(run_dir, "model.pt")
        agent.save(model_path)
        snapshot_configs(run_dir)
        print(f"\n저장 완료: {model_path}")
        print(f"재생 방법:      python test.py "
              f"{os.path.relpath(model_path, BASE)}")
        print(f"학습 곡선 보기: tensorboard --logdir {RESULTS_DIR}"
              f"  → http://localhost:6006")
