"""SUMO 고속도로 RL 환경 — Gymnasium 인터페이스 + TraCI.

┌─────────────────────────────────────────────────────────────┐
│  강화학습 루프에서 이 클래스의 역할                              │
│                                                             │
│    obs, _ = env.reset()          # 에피소드 시작, 초기 관측     │
│    obs, r, term, trunc, _ = env.step(action)  # 1스텝 진행    │
│                                                             │
│  내부적으로는 TraCI(Traffic Control Interface)라는 소켓 API로  │
│  SUMO 시뮬레이터를 원격 조종한다:                               │
│    traci.simulationStep()  → 시뮬레이션을 1스텝 진행            │
│    traci.vehicle.setSpeed() → 차량 속도 명령                   │
│    traci.vehicle.getSpeed() 등 → 상태 조회                     │
└─────────────────────────────────────────────────────────────┘

관측/행동/보상의 구체적인 정의는 configs/mdp_config.py 에 있고,
이 파일은 그 설정을 받아서 실행만 한다. (MDP를 바꿀 때 이 파일 수정 불필요)
"""
import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# SUMO 설치 폴더의 tools/ 안에 traci, sumolib 파이썬 모듈이 들어있다.
# pip로도 설치했지만, 버전 불일치를 피하려면 SUMO_HOME 쪽을 우선 경로에 추가.
#
# SUMO_HOME이 설정 안 돼 있으면 OS별 기본 설치 경로를 순서대로 탐색해서
# 자동으로 잡아준다 (환경변수 설정을 깜빡해도 동작하도록).
if "SUMO_HOME" not in os.environ:
    _candidates = [
        # macOS (Homebrew) — Apple Silicon / Intel
        "/opt/homebrew/opt/sumo/share/sumo",
        "/usr/local/opt/sumo/share/sumo",
        # Linux (apt)
        "/usr/share/sumo",
        # Windows (기본 인스톨러 경로)
        r"C:\Program Files (x86)\Eclipse\Sumo",
        r"C:\Program Files\Eclipse\Sumo",
    ]
    for _p in _candidates:
        if os.path.isdir(_p):
            os.environ["SUMO_HOME"] = _p
            print(f"[sumo_env] SUMO_HOME 자동 감지: {_p}")
            break
    else:
        raise RuntimeError(
            "SUMO_HOME 환경변수가 없고 기본 경로에서도 SUMO를 찾지 못했습니다.\n"
            "  Windows: 인스톨러 재설치(SUMO_HOME 체크) 또는 환경변수 수동 설정\n"
            "  macOS  : ./setup_mac.sh 실행 또는 README_MAC.md 참고\n"
            "  Linux  : sudo apt install sumo sumo-tools 후 "
            "export SUMO_HOME=/usr/share/sumo")

sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import traci
from sumolib import checkBinary

from env.road_builder import npc_vtype_ids, osm_config, prefill_plan

# UN R157 5.2.3.3항 최소 시간 간격표: 속도(m/s) → t_front(s)
_R157_SPEEDS = (2.0, 2.78, 5.56, 8.33, 11.11, 13.89, 16.67)
_R157_TIMES = (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6)

# SUMO 신호 상태 문자 → 관측값 (적색 계열 -1, 황색 0.5, 녹색 1, 꺼짐 0)
_SIGNAL_VALUE = {"r": -1.0, "R": -1.0, "s": -1.0, "u": -1.0,
                 "y": 0.5, "Y": 0.5, "g": 1.0, "G": 1.0}


class SumoHighwayEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg_path, road, ego, mdp_sim, mdp_obs, action, reward,
                 traffic=None, gui=False, gui_autostart=True, training=False):
        """
        Args:
            cfg_path: road_builder.build()가 만든 .sumocfg 경로
            road:     configs.road_config.ROAD
            ego:      configs.road_config.EGO (가속/감속 한계가 여기서 옴)
            mdp_sim:  configs.mdp_config.SIMULATION
            mdp_obs:  configs.mdp_config.OBSERVATION
            action:   configs.mdp_config.ACTION (행동 차원 수)
            reward:   configs.mdp_config.REWARD
            traffic:  configs.road_config.TRAFFIC (프리필/컨트롤러 정보).
                None이면 프리필 없이 flow만으로 동작 (하위호환)
            gui:      True면 sumo-gui(화면 있는 버전)로 실행
            gui_autostart: (gui=True일 때만 의미)
                True  → 창이 뜨면 시뮬레이션이 자동으로 시작
                False → 사용자가 ▶(플레이) 버튼을 누를 때까지 대기.
                        sumo-gui는 --start 옵션이 없으면 TraCI의
                        simulationStep() 호출도 플레이 전까지 블록되므로,
                        에피소드 전체가 "▶를 눌러야 시작"되는 효과가 난다.
            training: True면 TRAFFIC["emergency_decel_curriculum"]에 따라 배경차
                비상제동 한계를 학습 진행에 맞춰 낮춘다. train.py만 True로 만든다.
                평가·데이터 수집은 False라 항상 최종값(TRAFFIC["emergency_decel"])을 쓴다.
        """
        super().__init__()
        self.cfg_path = cfg_path
        self.gui = gui
        self.gui_autostart = gui_autostart

        # ----- 도로/차량 설정에서 필요한 값들 -----
        # OSM 도로망: 구간마다 제한속도가 다르고 ego 경로가 여러 개다
        osm = osm_config(road)
        self.osm = osm
        self.vmax = float((osm or road)["speed_limit"])   # 속도 정규화 기준
        self.ego_routes = list(osm["ego_routes"]) if osm else ["r0"]
        self.num_lanes = int(road["num_lanes"])
        # 시작 차선: "center"면 중앙 차선(차선수//2)으로 자동 결정.
        #   3차선 → 1번(가운데), 5차선 → 2번. 차선 수를 바꿔도 항상 가운데라
        #   좌/우 양쪽으로 변경 여지가 대칭으로 생긴다 (탐험에 유리).
        _dl = ego["depart_lane"]
        if str(_dl).lower() == "random":
            # 출발 구간의 차선 중 하나를 무작위로 고른다. 경로와 무관하게 고르므로
            # 진출 차선까지 여러 번 차선을 바꿔야 하는 출발도 생긴다.
            # 경로별로 남은 거리 안에 차선을 다 바꿀 수 없는 차선은 첫 투입 뒤 제외한다.
            self.ego_depart_lane = "random"
        elif osm:
            # 경로마다 차선 수가 달라서, 경로를 따라 차선 변경이 가장 적게 필요한
            # 차선으로 출발한다 ("free"는 출발 구간이 짧으면 따라갈 수 없는 차선을
            # 골라 SUMO가 투입을 거부할 수 있다)
            self.ego_depart_lane = "best"
        elif str(_dl).lower() == "center":
            self.ego_depart_lane = str(int(road["num_lanes"]) // 2)
        else:
            self.ego_depart_lane = str(_dl)
        # ---- 프리필(도로 전체 사전 배치) 설정 ----
        self.traffic = traffic or {}
        pf = (self.traffic.get("prefill") or {})
        self.prefill_enabled = bool(pf.get("enabled", False))
        self.prefill_start = float(pf.get("start_offset", 80.0))
        self.prefill_end_margin = float(pf.get("end_margin", 50.0))
        self.prefill_jitter = float(pf.get("spacing_jitter", 0.3))
        self.road_length = float(road["length"])
        self.traffic_max_speed = float(self.traffic.get("max_speed", self.vmax))
        # 차선당 배치 간격: 지정값이 없으면 교통류 이론대로 자동 계산
        #   차선당 유량 q_lane = (vehs/h) / 3600 / 차선수  [대/초]
        #   간격 = 속도 / q_lane   (그 속도로 흐를 때 차 사이 평균 거리)
        self.prefill_fixed_spacing = float(pf["spacing"]) if pf.get("spacing") else None
        # 프리필 배치 계획: 직선 도로면 r0 한 구간, 사거리면 모든 엣지 구간
        # (road_builder.prefill_plan 참고 — 구간마다 경로/길이/차선/교통량)
        self.prefill_plan = prefill_plan(road, self.traffic) if traffic else []
        # 컨트롤러 타입 샘플링 준비 (rou.xml의 vType과 대응. minGap 단계마다 vType이 나뉜다)
        ctrls = self.traffic.get("controllers") or {}
        if ctrls:
            self._prefill_types, probs = [], []
            for n, c in ctrls.items():
                ids = npc_vtype_ids(n, self.traffic)
                self._prefill_types += ids
                probs += [float(c.get("probability", 1.0)) / len(ids)] * len(ids)
            probs = np.array(probs)
            self._prefill_probs = probs / probs.sum()
        else:
            self._prefill_types, self._prefill_probs = ["car"], np.array([1.0])
        self.ego_depart_speed = str(ego["depart_speed"])
        self.depart_lane_spacing = float(ego.get("depart_lane_spacing", 0.0))
        self.depart_lanes: dict[str, list[int]] = {}
        # 행동 a ∈ [-1,1] 을 실제 가속도로 변환할 때 쓰는 스케일:
        #   a = +1 → +max_accel (풀 액셀),  a = -1 → -max_decel (풀 브레이크)
        self.max_accel = float(ego["accel"])   # m/s^2
        self.max_decel = float(ego["decel"])   # m/s^2 (양수로 저장, 부호는 계산 시)
        # 기동 횡방향 행동 a ∈ [-1,1] → 횡가속도 a × max_lat_accel.
        #   횡속도는 이 가속도를 매 스텝 적분하고 ±max_speed_lat로 제한한다.
        self.max_lat_accel = float(ego["lc_accel_lat"])   # m/s^2
        self.max_speed_lat = float(ego["max_speed_lat"])  # m/s
        # 횡속도 상한을 종방향 속도에 비례시킨다: |횡속도| ≤ min(max_speed_lat, 이 값 × 속도).
        #   실제 차는 앞으로 가야 옆으로 움직일 수 있다. 정지 상태에서는 옆으로 움직이지 않는다.
        self.lat_speed_per_speed = float(ego.get("lat_speed_per_speed", 0.2))
        # 횡속도는 왼쪽이 +이다.
        self.lat_speed = 0.0
        # 걸쳤다가 취소한 뒤 다시 트리거할 수 없는 시간 (스텝)
        self.lc_cooldown_steps = int(round(
            float(action.get("retrigger_cooldown", 2.0)) / float(mdp_sim["step_length"])))
        self.lc_block_until = 0
        self._reset_maneuver()

        # ---- 배경차 비상제동 커리큘럼 (training=True일 때만) ----
        #   학습 초반에는 배경차가 강하게 제동해 차선변경 시도가 곧바로 충돌로 끝나지 않게 하고,
        #   steps 동안 최종값(emergency_decel)까지 선형으로 낮춘다.
        self.training = bool(training)
        self.total_steps = 0
        self.npc_decel_final = float(self.traffic_cfg_value(traffic, "emergency_decel", 9.0))
        self.npc_decel_curriculum = (dict((traffic or {}).get("emergency_decel_curriculum") or {})
                                     if self.training else {})
        self.npc_emergency_decel = self.npc_decel_final

        # ----- MDP 설정 -----
        self.step_length = float(mdp_sim["step_length"])
        self.max_steps = int(mdp_sim["max_steps"])
        # 경로별 시간 제한(초). reset()에서 고른 경로의 제한을 episode_max_steps로 바꾼다.
        self.route_timeout = dict(osm.get("ego_route_timeout", {})) if osm else {}
        self.episode_max_steps = self.max_steps
        self.warmup_steps = int(osm["warmup_steps"] if osm else mdp_sim["warmup_steps"])
        self.visibility = float(mdp_obs["visibility"])   # 관측 거리 W
        # 밀도 정규화용 "차량 1대가 점유하는 길이" (차체 + 최소 간격)
        self.density_veh_len = float(mdp_obs.get("density_veh_length", 7.5))
        # 추가 컬럼 정규화 기준: 횡속도(m/s), 경로 잔여거리(m)
        self.lat_speed_scale = float(mdp_obs.get("lat_speed_scale", 1.0))
        self.route_horizon = float(mdp_obs.get("route_horizon", 500.0))
        self.signal_horizon = float(mdp_obs.get("signal_horizon", 100.0))
        self.junction_horizon = float(mdp_obs.get("junction_horizon", 100.0))
        self.straddle_horizon = float(mdp_obs.get("straddle_horizon", 10.0))
        self.lane_offset_scale = float(mdp_obs.get("lane_offset_scale", 6.0))
        self.act_dim = int(action["dim"])
        self.accel_index = int(action.get("accel_index", 0))
        self.lane_change_index = int(action.get("lane_change_index", 1))
        self.lat_accel_index = int(action.get("lat_accel_index", 2))
        self.lane_change_threshold = float(action.get("lane_change_threshold", 0.5))
        self.rw = reward

        # GUI는 ego를 따라가며 가까이 보이도록 기본값을 둔다.
        self.gui_zoom = float(mdp_sim.get("gui_zoom", 1200.0))
        self.gui_delay = float(mdp_sim.get("gui_delay", 100.0))
        self.gui_track_ego = bool(mdp_sim.get("gui_track_ego", True))

        self.ego_id = "ego"        # SUMO 안에서 ego 차량을 식별하는 ID
        self.step_count = 0
        self._nav_phi = 0.0
        self._started = False      # SUMO 프로세스가 떠 있는지 여부

        # Gymnasium 행동 공간:
        #   action[0] = 종방향 가감속 raw ∈ [-1,1]
        #   action[1] = 차선변경 트리거 raw ∈ [-1,1] (왼쪽 +)
        #   action[2] = 기동 중 횡가속도 raw ∈ [-1,1] (기동 방향 +)
        # 완료된 차선변경 방향은 info["lane_change"]에 -1/0/+1로 반환한다.
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(self.act_dim,), dtype=np.float32)
        # 관측: 46차원, 범위 [-1,1]. 레이아웃은 env/mdp_config.py 주석 참고.
        #   시야 차선: 좌2, 좌1, 현재, 우1, 우2 (5개) — 3차선 도로에서
        #   맨 왼쪽에 있어도 오프셋 -2로 맨 오른쪽 차선까지 보인다.
        self.obs_offsets = (+2, +1, 0, -1, -2)
        n_lanes_view = len(self.obs_offsets)
        # 추가 컬럼 15개는 인접 선행차 횡정보 4, ego 횡상태 2, 인접 후행차 가속도 2,
        # 내비게이션 2, 신호 2, 기동 방향 1, 교차로 거리 1, 걸친 후 경과시간 1이다.
        self.obs_dim = 1 + n_lanes_view * 4 + n_lanes_view + n_lanes_view + 15
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(self.obs_dim,),
                                            dtype=np.float32)

    @staticmethod
    def traffic_cfg_value(traffic, key, default):
        """TRAFFIC 설정에서 ``key`` 값을 읽는다. 설정이 없으면 ``default``."""
        return (traffic or {}).get(key, default)

    def _apply_npc_emergency_decel(self) -> None:
        """커리큘럼에 따른 배경차 비상제동 한계를 모든 배경차 vType에 적용한다."""
        cur = self.npc_decel_curriculum
        if cur:
            frac = min(self.total_steps / max(float(cur.get("steps", 1)), 1.0), 1.0)
            start = float(cur.get("start", 9.0))
            self.npc_emergency_decel = start + (self.npc_decel_final - start) * frac
        else:
            self.npc_emergency_decel = self.npc_decel_final
        for vtype in self._prefill_types:
            try:
                traci.vehicletype.setEmergencyDecel(vtype, self.npc_emergency_decel)
            except traci.TraCIException:
                pass

    # ==================================================================
    # SUMO 프로세스 시작
    # ==================================================================
    def _start_sumo(self):
        # gui=True면 화면이 있는 sumo-gui, 아니면 화면 없는 sumo (학습용, 훨씬 빠름)
        binary = checkBinary("sumo-gui" if self.gui else "sumo")
        cmd = [
            binary, "-c", self.cfg_path,
            "--step-length", str(self.step_length),
            # 충돌한 차량을 시뮬레이션에서 제거. (기본값 teleport는 순간이동시켜버려서
            # RL 학습에는 부적합. remove로 해야 "충돌 → 에피소드 종료"가 깔끔함)
            "--collision.action", "remove",
            # 교차로 내부(junction)에서의 충돌도 감지 (기본값은 감지 안 함)
            "--collision.check-junctions", "true",
            "--no-warnings", "true",   # 콘솔 경고 끄기
            "--no-step-log", "true",   # 스텝 로그 끄기
            # 매 에피소드 다른 시드 → 배경 교통 패턴이 달라져서 과적합 방지
            "--seed", str(np.random.randint(0, 100000)),
        ]
        if self.gui:
            cmd += ["--quit-on-end"]      # 에피소드 종료 시 창 정리
            cmd += ["--delay", str(self.gui_delay)]
            if self.gui_autostart:
                cmd += ["--start"]        # 자동 재생
            # --start가 없으면: 창이 뜬 뒤 ▶(플레이)를 눌러야 진행된다.
            #   매 에피소드가 reset()에서 SUMO를 새로 띄우므로,
            #   에피소드마다 ▶를 눌러 시작하게 된다.
        traci.start(cmd)   # SUMO를 자식 프로세스로 띄우고 소켓 연결
        self._started = True

    # ==================================================================
    # 에피소드 초기화
    # ==================================================================
    def _prefill_traffic(self):
        """도로 전 구간에 배경차를 일정 간격(±jitter)으로 미리 배치한다.

        flow는 도로 입구에서만 차를 만들기 때문에, 프리필이 없으면
        에피소드 초반의 긴 도로가 텅 비어 "빈 도로 주행"만 잔뜩 학습된다.
        프리필은 첫 스텝부터 ego가 실제 교통류 한가운데서 시작하게 만든다.

        - 위치: 구간(prefill_plan)마다 start ~ (구간 길이 - end_margin), 차선별 독립 배치
          (사거리면 접근로/진출로/교차도로 모든 엣지에 배치. ego 출발 구간만
           start_offset만큼 비움)
        - 간격: prefill_spacing × (1 ± jitter) 무작위
        - 타입: controllers 확률 비율로 샘플링 (rou.xml의 vType과 동일)
        - 속도: departSpeed="max" → 각 개체가 "자기 허용 최고속도"로 시작.
          컨트롤러별 max_speed/speedDev 차등이 프리필에도 그대로 반영되고,
          개체별 무작위 speedFactor 때문에 숫자 속도를 주면 SUMO가
          "허용속도 초과"로 거부할 수 있는 문제도 함께 피한다.
        배치된 차들은 SUMO의 각자 컨트롤러(IDM 등)가 정상적으로 운전한다.
        """
        rng = np.random.default_rng()
        count = 0
        # 차선별 간격 배율 (없으면 전 차선 동일). 인덱스 0 = 맨 오른쪽.
        factors = (self.traffic.get("prefill", {})
                   .get("lane_spacing_factor")
                   or [1.0] * self.num_lanes)
        for seg in self.prefill_plan:
            if seg["vph"] <= 0:
                continue
            # 구간 배치 간격: 지정값이 없으면 교통류 이론대로 자동 계산
            #   차선당 유량 q_lane = (vehs/h) / 3600 / 차선수  [대/초]
            #   간격 = 속도 / q_lane   (그 속도로 흐를 때 차 사이 평균 거리)
            if self.prefill_fixed_spacing:
                spacing = self.prefill_fixed_spacing
            else:
                q_lane = max(seg["vph"] / 3600.0 / max(len(seg["lanes"]), 1), 1e-6)
                spacing = self.traffic_max_speed / q_lane
            # ego 출발 구간만 시작부를 넉넉히 비우고, 나머지 구간은 조금만 띄운다
            start = self.prefill_start if seg["ego_start"] else 10.0
            # 차선 배율은 "직진 차로 순서" 기준 (회전 포켓 차선은 plan에서 제외됨)
            for k, lane in enumerate(seg["lanes"]):
                f = float(factors[k]) if k < len(factors) else 1.0
                lane_spacing = spacing * f
                pos = start + rng.uniform(0, lane_spacing)
                while pos < seg["length"] - self.prefill_end_margin:
                    vid = f"pre_{lane}_{count}"
                    vtype = str(rng.choice(self._prefill_types, p=self._prefill_probs))
                    try:
                        traci.vehicle.add(
                            vid, seg["route"], typeID=vtype,
                            departLane=str(lane),
                            departPos=f"{pos:.1f}",
                            departSpeed="max")
                        count += 1
                    except traci.TraCIException:
                        pass  # 겹침 등으로 실패한 자리는 건너뜀
                    step = lane_spacing * rng.uniform(
                        1.0 - self.prefill_jitter, 1.0 + self.prefill_jitter)
                    pos += max(step, 10.0)
        return count

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # SUMO 상태를 완전히 초기화하는 가장 확실한 방법은 껐다 켜는 것.
        if self._started:
            traci.close()
            self._started = False
        self._start_sumo()
        if self.npc_decel_curriculum:
            self._apply_npc_emergency_decel()

        # 도로 전 구간에 배경차를 미리 배치 (첫 스텝부터 실제 교통류 상태)
        if self.prefill_enabled:
            self._prefill_traffic()
        # 배치된 차들이 자리를 잡도록 몇 스텝만 정착시킨다.
        for _ in range(self.warmup_steps):
            traci.simulationStep()

        # ego 차량 투입 — 반드시 "도로 시작점(0m)"에서 출발
        route = str(np.random.choice(self.ego_routes))
        if route in self.route_timeout:
            self.episode_max_steps = int(np.ceil(
                self.route_timeout[route] / self.step_length))
        else:
            self.episode_max_steps = self.max_steps
        depart_lane = self.ego_depart_lane
        if depart_lane == "random":
            first_edge = traci.route.getEdges(route)[0]
            lanes = self.depart_lanes.get(
                route, list(range(traci.edge.getLaneNumber(first_edge))))
            depart_lane = str(np.random.choice(lanes))
        traci.vehicle.add(self.ego_id, route, typeID="ego_type",
                          departLane=depart_lane,
                          departPos="0",
                          departSpeed=self.ego_depart_speed)

        # ★ 중요: speedMode=0 → SUMO의 내장 안전장치(자동 안전제동 등)를 전부 끔.
        #   이걸 꺼야 RL이 속도를 온전히 제어하고, 잘못 배우면 실제로 충돌한다.
        #   (켜두면 SUMO가 알아서 브레이크를 밟아줘서 충돌이 거의 안 나고,
        #    그러면 "충돌 회피"라는 학습 신호 자체가 사라진다)
        traci.vehicle.setSpeedMode(self.ego_id, 0)
        # SUMO의 자율 차선변경은 끄고, RL이 내리는 TraCI 차선변경 명령만 사용한다.
        traci.vehicle.setLaneChangeMode(self.ego_id, 0)
        # 횡가속도 제한은 _apply_lateral이 직접 적용한다. SUMO가 매 스텝의
        # changeSublane 명령마다 자체 가감속 곡선을 다시 그리지 않도록
        # SUMO 쪽 횡가속도 한계는 충분히 크게 둔다.
        traci.vehicle.setParameter(self.ego_id, "laneChangeModel.lcAccelLat", "100")

        traci.simulationStep()  # ego가 실제로 도로에 나타나도록 1스텝 진행

        # ── ego 삽입 보증 ──
        # 도로 입구는 flow가 차를 스폰하는 자리라, 시작 차선 입구가 잠시
        # 막혀 있으면 SUMO가 삽입을 "지연"시킨다. 그 경우 ego 없이
        # 에피소드가 진행되어 즉시 종료되는 버그가 되므로, 삽입될 때까지
        # (최대 30스텝) 기다린다. 지연되어도 요청한 차선은 유지된다.
        _wait = 0
        while self.ego_id not in traci.vehicle.getIDList() and _wait < 30:
            traci.simulationStep()
            _wait += 1
        if self.ego_id not in traci.vehicle.getIDList():
            raise RuntimeError(
                "ego 삽입 실패: 도로 입구가 30스텝 이상 막혀 있습니다. "
                "TRAFFIC['vehs_per_hour']를 낮추거나 depart_speed를 확인하세요.")
        if self.ego_depart_lane == "random" and route not in self.depart_lanes:
            self.depart_lanes[route] = self._feasible_depart_lanes()

        # GUI 기본 시점: ego를 화면 중앙에서 추적하고 충분히 가까이 확대한다.
        # 긴 직선도로 전체를 한 화면에 맞추는 기본 뷰는 차량이 점/삼각형처럼
        # 보이기 쉬우므로 학습/평가 관찰용으로는 이 설정이 더 직관적이다.
        if self.gui and self.ego_id in traci.vehicle.getIDList():
            try:
                view_ids = traci.gui.getIDList()
                if view_ids:
                    view_id = view_ids[0]
                    # tracking을 먼저 켜고 zoom을 마지막에 적용해야 SUMO가
                    # 차량 추적을 시작하면서 viewport를 다시 덮어쓰지 않는다.
                    if self.gui_track_ego:
                        traci.gui.trackVehicle(view_id, self.ego_id)
                    traci.gui.setZoom(view_id, self.gui_zoom)
            except traci.TraCIException:
                pass

        self.step_count = 0
        self.lat_speed = 0.0
        self.lc_block_until = 0
        self._reset_maneuver()
        self._nav_phi = (0.0 if traci.vehicle.getRoadID(self.ego_id).startswith(":")
                         else self._nav_potential())
        return self._get_obs(), {}

    def _reset_maneuver(self) -> None:
        """차선변경 기동 상태를 기동 없음으로 되돌린다."""
        # 기동 방향이다. 0은 기동 없음, +1은 왼쪽, -1은 오른쪽이다.
        self.lc_dir = 0
        self.lc_target = -1
        self.lc_edge = ""
        # 기동 시작 후 차체가 한 번이라도 차선 경계에 걸쳤는지 나타낸다.
        self.lc_straddled = False
        # 처음 걸친 뒤 지난 스텝 수. 걸치기 전과 기동이 없을 때는 0이다.
        self.lc_straddle_steps = 0

    # ==================================================================
    # 관측 만들기 (레이아웃은 env/mdp_config.py 주석 참고)
    #
    # 핵심 설계: 특정 도로에 대한 하드코딩 없이, 매 스텝 SUMO 네트워크를
    # 실시간 조회해서 "지금 내 주변"을 계산한다. 그래서 도로 구조가
    # 어떻게 바뀌어도(차선 수, 합류, 차선 감소 등) 이 코드는 그대로 동작한다.
    #
    # 사용하는 TraCI API:
    #   getLeader / getFollower : 같은 차선의 앞/뒤 차량 (edge 경계도 넘어 탐색)
    #   getNeighbors(mode)      : 옆 차선의 앞/뒤 차량.
    #       mode 비트: bit0 = 0:왼쪽 / 1:오른쪽,  bit1 = 0:후행 / 1:선행
    #   lane.getLinks           : 차선이 다음 edge로 이어지는지 (연결성 판단)
    # ==================================================================

    # (차선 오프셋, 선행차 getNeighbors mode, 후행차 mode)
    #   오프셋 +1 = 왼쪽 차선 (SUMO는 차선 index가 왼쪽으로 갈수록 커짐)
    _NEIGHBOR_MODES = {+1: (0b10, 0b00),   # 왼쪽: 선행=2, 후행=0
                       -1: (0b11, 0b01)}   # 오른쪽: 선행=3, 후행=1

    def _neighbor(self, offset: int, leader: bool):
        """오프셋 차선(-1=오른쪽, 0=현재, +1=왼쪽)의 가장 가까운
        선행(leader=True)/후행 차량을 (거리 m, 그 차량 속도)로 반환.
        visibility 밖이거나 없으면 None."""
        if offset == 0:
            if leader:
                res = traci.vehicle.getLeader(self.ego_id, self.visibility)
                if res is None:
                    return None
                vid, gap = res
            else:
                vid, gap = traci.vehicle.getFollower(self.ego_id, self.visibility)
                if vid == "":
                    return None
        elif abs(offset) >= 2:
            # 두 칸 이상 옆 차선: SUMO getNeighbors는 "바로 옆"만 지원하므로
            # 차선 직접 스캔(_scan_lane_neighbors — 같은 edge의 차량 목록을
            # 훑어 가장 가까운 앞/뒤를 찾음)으로 감지한다.
            front, rear = self._scan_lane_neighbors(offset)
            res = front if leader else rear
            if res is None or res[0] < 0 or res[0] > self.visibility:
                return None
            return float(res[0]), float(res[1])
        else:
            mode_leader, mode_follower = self._NEIGHBOR_MODES[offset]
            pairs = traci.vehicle.getNeighbors(
                self.ego_id, mode_leader if leader else mode_follower)
            if not pairs:
                return None
            vid, gap = min(pairs, key=lambda p: p[1])   # 가장 가까운 차량

        if gap < 0 or gap > self.visibility:
            return None
        return float(gap), traci.vehicle.getSpeed(vid)

    def _lane_connectivity(self, offset: int) -> float:
        """오프셋 차선의 '끊김 정도'를 반환.

        -1 : 그 차선이 존재하지 않음
         1 : visibility 이내에서 계속 이어짐
        0~1: (차선이 끝나는 지점까지의 거리 / visibility)
             → 값이 작을수록 곧 차선이 사라진다는 경고

        일반화 원리: 차선의 남은 길이와, 차선 끝에서 다음 edge로의
        연결(link) 존재 여부를 SUMO 네트워크에서 직접 조회한다.
        차선 감소/합류 지점은 "링크 없는 차선의 끝"으로 자연히 감지된다.
        """
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            if edge.startswith(":"):          # 교차로(junction) 내부 통과 중
                return 1.0
            lane_index = traci.vehicle.getLaneIndex(self.ego_id) + offset
            if lane_index < 0 or lane_index >= traci.edge.getLaneNumber(edge):
                return -1.0                   # 그런 차선이 없음
            lane_id = f"{edge}_{lane_index}"
            remaining = (traci.lane.getLength(lane_id)
                         - traci.vehicle.getLanePosition(self.ego_id))
            if remaining > self.visibility:
                return 1.0                    # 시야 내에선 계속 이어짐
            if self._route_links(lane_id):
                return 1.0                    # edge가 끝나도 경로의 다음 edge로 연결됨
            return float(np.clip(remaining / self.visibility, 0.0, 1.0))
        except traci.TraCIException:
            return 1.0                        # 조회 실패 시 중립값

    def _route_links(self, lane_id: str):
        """lane_id의 연결(link) 중 ego 경로의 다음 edge로 이어지는 것만 반환.

        사거리의 회전 포켓 차선은 다른 방향(좌/우회전)으로는 이어지지만
        ego의 직진 경로로는 이어지지 않으므로 "곧 끊기는 차선"으로 취급된다.
        경로의 마지막 edge(도로 끝)라면 빈 리스트.
        """
        route = traci.vehicle.getRoute(self.ego_id)
        idx = traci.vehicle.getRouteIndex(self.ego_id)
        if idx < 0 or idx + 1 >= len(route):
            return []
        nxt = route[idx + 1]
        return [l for l in traci.lane.getLinks(lane_id)
                if traci.lane.getEdgeID(l[0]) == nxt]

    def _lane_density(self, offset: int) -> float:
        """오프셋 차선의 '전방 차량 밀도'를 0~1로 반환.

        밀도 = (시야 W 이내 전방 차량 수) / (그 구간의 최대 수용 대수)
               최대 수용 대수 = 사용가능거리 / density_veh_length

        인코딩 관례 (막힘 정도로 읽히도록):
            1.0 : 차선이 없거나 사실상 진입 불가  ← "꽉 찬 것"처럼 취급해서
                  에이전트가 자연스럽게 회피하게 만든다
            0.0 : 텅 빈 차선
        일반화: 차선이 시야보다 짧으면 link를 따라 다음 차선까지 이어서 센다.
        곧 끊기는 차선(link 없음)은 남은 거리만큼만 세므로 분모가 줄어
        같은 대수여도 밀도가 높게 나온다 → "끊기는 차선에 차가 몰려있음" 신호.
        """
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            if edge.startswith(":"):          # 교차로 내부: 중립값
                return 0.0
            lane_index = traci.vehicle.getLaneIndex(self.ego_id) + offset
            if lane_index < 0 or lane_index >= traci.edge.getLaneNumber(edge):
                return 1.0                    # 차선 없음 = 꽉 참 취급
            lane_id = f"{edge}_{lane_index}"
            ego_pos = traci.vehicle.getLanePosition(self.ego_id)
            remaining = traci.lane.getLength(lane_id) - ego_pos
            links = self._route_links(lane_id)

            # 이 차선에서 앞으로 실제로 쓸 수 있는 거리
            usable = self.visibility if links else min(self.visibility, remaining)
            if usable <= 0:
                return 1.0

            count = 0
            # (a) 현재 차선 구간의 전방 차량
            for vid in traci.lane.getLastStepVehicleIDs(lane_id):
                if vid == self.ego_id:
                    continue
                d = traci.vehicle.getLanePosition(vid) - ego_pos
                if 0.0 < d <= min(self.visibility, remaining):
                    count += 1
            # (b) 시야가 edge 끝을 넘고 다음 차선으로 이어지면 이어서 센다
            if links and remaining < self.visibility:
                next_lane_id = links[0][0]    # 첫 번째 연결 차선
                for vid in traci.lane.getLastStepVehicleIDs(next_lane_id):
                    if traci.vehicle.getLanePosition(vid) <= \
                            self.visibility - remaining:
                        count += 1

            max_count = usable / self.density_veh_len
            return float(np.clip(count / max_count, 0.0, 1.0))
        except traci.TraCIException:
            return 0.0

    def _adjacent_id(self, offset: int, leader: bool):
        """좌1(+1)/우1(-1) 차선의 가장 가까운 선행/후행 차량 ID.
        _neighbor와 같은 기준(getNeighbors, 시야 W 이내)으로 고른다."""
        mode_leader, mode_follower = self._NEIGHBOR_MODES[offset]
        pairs = [p for p in traci.vehicle.getNeighbors(
                     self.ego_id, mode_leader if leader else mode_follower)
                 if 0 <= p[1] <= self.visibility]
        if not pairs:
            return None
        return min(pairs, key=lambda p: p[1])[0]

    def _extra_obs(self):
        """추가 컬럼 14개 (레이아웃은 env/mdp_config.py 주석 참고).

        SUMO의 횡위치는 차선 중앙 기준이며 왼쪽이 +이다.
        """
        obs = []
        # [31..34] 좌1, 우1 선행차: 내 차선 경계까지 횡거리, 내 쪽 횡속도
        for offset in (+1, -1):
            vid = self._adjacent_id(offset, leader=True)
            if vid is None:
                obs += [1.0, -1.0]                 # 없음: 멀고 + 위협 없음
                continue
            lane_w = traci.lane.getWidth(traci.vehicle.getLaneID(vid))
            half_w = traci.vehicle.getWidth(vid) / 2.0
            lat = traci.vehicle.getLateralLanePosition(vid)
            lat_v = traci.vehicle.getLateralSpeed(vid)
            if offset == +1:   # 왼쪽 차: 그 차선의 오른쪽 경계가 내 차선 경계
                dist, toward = lat - half_w + lane_w / 2.0, -lat_v
            else:              # 오른쪽 차: 그 차선의 왼쪽 경계가 내 차선 경계
                dist, toward = lane_w / 2.0 - (lat + half_w), lat_v
            obs += [dist / lane_w, toward / self.lat_speed_scale]

        # [35, 36] ego 차선 중앙 대비 횡위치, 횡속도
        lane_id = traci.vehicle.getLaneID(self.ego_id)
        lane_w = traci.lane.getWidth(lane_id)
        obs.append(traci.vehicle.getLateralLanePosition(self.ego_id) / (lane_w / 2.0))
        obs.append(traci.vehicle.getLateralSpeed(self.ego_id) / self.lat_speed_scale)

        # [37, 38] 좌1, 우1 후행차 가속도 (없으면 0)
        for offset in (+1, -1):
            vid = self._adjacent_id(offset, leader=False)
            obs.append(0.0 if vid is None
                       else traci.vehicle.getAcceleration(vid) / self.max_decel)

        # [39, 40] 내비게이션: 필요한 차선 변경 수(+왼쪽), 현재 차선 잔여거리.
        #   getBestLanes의 length는 차선 시작점 기준이라 내 위치를 뺀다.
        #   교차로 내부 등 현재 차선이 목록에 없으면 "변경 불필요, 멀리 남음".
        best_offset, remain = self._best_lane_info()
        obs += [best_offset / self.lane_offset_scale, max(remain, 0.0) / self.route_horizon]

        # [41, 42] 다음 신호: 상태(적 -1, 황 0.5, 녹 1, 없음 0), 정지선까지 거리
        #   signal_horizon 밖의 신호는 보이지 않는 것으로 보고 "신호 없음"으로 둔다.
        light = self._next_light()
        if light is None or light[2] > self.signal_horizon:
            obs += [0.0, 1.0]
        else:
            obs += [_SIGNAL_VALUE.get(light[3], 0.0), light[2] / self.signal_horizon]

        # [43, 44] 차선변경 기동 방향, 다음 교차로까지 거리
        obs.append(float(self.lc_dir))
        obs.append(min(self._junction_distance(), self.junction_horizon)
                   / self.junction_horizon)

        # [45] 처음 걸친 뒤 경과시간 / straddle_horizon (걸치지 않았으면 0)
        obs.append(min(self.lc_straddle_steps * self.step_length / self.straddle_horizon, 1.0))
        return obs

    def _best_lane_info(self) -> tuple[float, float]:
        """현재 차선의 (필요한 차선 변경 수, 경로를 따라 주행 가능한 잔여 거리 m)를 반환한다.

        차선 변경 수는 왼쪽이 +다. 교차로 내부처럼 현재 차선이 ``getBestLanes``
        목록에 없으면 (0, ``route_horizon``)을 반환한다.
        """
        lane_id = traci.vehicle.getLaneID(self.ego_id)
        for lane in traci.vehicle.getBestLanes(self.ego_id):
            if lane[0] == lane_id:
                return (float(lane[3]),
                        lane[1] - traci.vehicle.getLanePosition(self.ego_id))
        return 0.0, self.route_horizon

    def _feasible_depart_lanes(self) -> list[int]:
        """현재 edge에서 남은 거리 안에 필요한 차선 변경을 마칠 수 있는 차선 번호를 반환한다.

        차선 변경 1회에 ``EGO["depart_lane_spacing"]`` m가 든다고 보고,
        |필요한 차선 변경 수| × spacing이 그 차선의 잔여 거리 이하인 차선만 남긴다.
        조건을 만족하는 차선이 없으면 변경이 필요 없는 차선을 반환한다.
        """
        edge = traci.vehicle.getRoadID(self.ego_id)
        pos = traci.vehicle.getLanePosition(self.ego_id)
        lanes = [(int(lane[0].rsplit("_", 1)[1]), abs(lane[3]), lane[1] - pos)
                 for lane in traci.vehicle.getBestLanes(self.ego_id)
                 if lane[0].rsplit("_", 1)[0] == edge]
        feasible = [index for index, offset, remain in lanes
                    if offset * self.depart_lane_spacing <= remain]
        return feasible or [index for index, offset, _ in lanes if offset == 0]

    def _exit_missed(self) -> bool:
        """경로로 이어지지 않는 차선의 끝에 도달했는지 반환한다.

        필요한 차선 변경 수가 0이 아니고 잔여 거리가 ``exit_miss_distance`` 이하이면
        출구를 놓친 것으로 본다. 이 차선에서는 경로를 따라 더 갈 수 없다.
        """
        offset, remain = self._best_lane_info()
        return offset != 0.0 and remain <= float(self.rw.get("exit_miss_distance", 5.0))

    def _nav_potential(self) -> float:
        """현재 차선의 경로 정합 포텐셜 Φ(s) ≤ 0을 반환한다.

        Φ(s) = -nav_shaping_weight × |best_offset| × clip(1 - remain / nav_shaping_horizon, 0, 1).
        출구가 horizon보다 멀면 0이고, 가까워질수록 비최적 차선의 포텐셜이 낮아진다.
        """
        weight = float(self.rw.get("nav_shaping_weight", 0.0))
        horizon = float(self.rw.get("nav_shaping_horizon", 0.0))
        if weight <= 0.0 or horizon <= 0.0:
            return 0.0
        offset, remain = self._best_lane_info()
        urgency = float(np.clip(1.0 - remain / horizon, 0.0, 1.0))
        return -weight * abs(offset) * urgency

    def _queue_cut_count(self) -> int:
        """ego 바로 뒤에 이어진 대기 줄의 차량 수를 반환한다.

        뒤차를 차례로 따라가며, 속도가 ``queue_speed`` 미만이고 앞차와의 간격이
        ``queue_gap`` 이하인 차량을 센다. 조건을 벗어나는 차량에서 멈춘다.
        """
        max_speed = float(self.rw.get("queue_speed", 5.0))
        max_gap = float(self.rw.get("queue_gap", 2.5))
        count, current = 0, self.ego_id
        while True:
            vid, gap = traci.vehicle.getFollower(current, max_gap)
            if (vid == "" or vid == self.ego_id or gap < 0.0 or gap > max_gap
                    or traci.vehicle.getSpeed(vid) >= max_speed):
                return count
            count += 1
            current = vid

    def _junction_distance(self) -> float:
        """ego에서 다음 교차로까지의 경로상 거리(m)를 반환한다.

        현재 edge의 끝을 교차로로 본다. 교차로 내부면 0이고, 경로의 마지막 edge면
        앞에 교차로가 없으므로 ``junction_horizon``을 반환한다.
        """
        if traci.vehicle.getRoadID(self.ego_id).startswith(":"):
            return 0.0
        route = traci.vehicle.getRoute(self.ego_id)
        if traci.vehicle.getRouteIndex(self.ego_id) + 1 >= len(route):
            return self.junction_horizon
        lane_len = traci.lane.getLength(traci.vehicle.getLaneID(self.ego_id))
        return max(lane_len - traci.vehicle.getLanePosition(self.ego_id), 0.0)

    def _next_light(self):
        """경로상 다음 신호 (tlsID, linkIndex, 거리, 상태문자). 없으면 None."""
        lights = traci.vehicle.getNextTLS(self.ego_id)
        return lights[0] if lights else None

    def _r157_min_gap(self, v: float) -> float:
        """속도 ``v``(m/s)에서의 최소 차간거리 d_min(m)을 반환한다.

        d_min = max(1.0 + 0.5 × v, v × t_front)이다. t_front는 UN R157 5.2.3.3항 표를
        선형보간하고, 60 km/h 초과는 1.6초로 유지된다. 2 m/s 이상에서는 R157과 같고,
        2 m/s 미만에서는 정지 시 1m에서 2 m/s일 때 2m까지 선형으로 늘어나 R157과 이어진다.
        """
        return max(1.0 + 0.5 * v, v * float(np.interp(v, _R157_SPEEDS, _R157_TIMES)))

    def _slow_penalty(self, ego_speed: float, leader, speed_ref: float) -> float:
        """앞 공간이 안전거리보다 넓은데 기준 속도보다 느리면 감점 크기를 반환한다.

        앞 장애물은 시야 안의 같은 차로 앞차와 적색·황색 신호 정지선 중 가까운 쪽이다.
        정지선은 속도 0인 앞차로 본다. 기준 속도는 장애물이 있으면 이 간격에서 낼 수 있는
        속도 min(``speed_ref``, sqrt(v_obs² + 2·b·max(간격 − d_min(v_obs), 0)))이고,
        없으면 ``speed_ref``다. b는 ``slow_comfort_decel``이다. 간격이 d_min이면 장애물
        속도와 같고, 간격이 넓을수록 기준 속도가 올라가 정지한 장애물 뒤에 멀리 서 있어도
        감점된다. ``leader``는
        :meth:`_neighbor`가 반환한 (minGap을 뺀 간격, 속도) 또는 None이다.

        기준 속도보다 ``slow_tolerance`` 비율까지 느린 것은 허용하고, 그보다 느린 만큼
        선형으로 감점한다. 정지 상태면 ``slow_penalty``를 모두 감점한다.
        """
        weight = float(self.rw.get("slow_penalty", 0.0))
        if weight <= 0.0:
            return 0.0
        obstacle = None
        if leader is not None:
            obstacle = (leader[0] + traci.vehicle.getMinGap(self.ego_id), leader[1])
        light = self._next_light()
        if (light is not None and light[2] <= self.visibility
                and _SIGNAL_VALUE.get(light[3], 0.0) in (-1.0, 0.5)
                and (obstacle is None or light[2] < obstacle[0])):
            obstacle = (light[2], 0.0)

        if obstacle is None:
            v_ref = speed_ref
        else:
            gap, v_obs = obstacle
            decel = float(self.rw.get("slow_comfort_decel", 2.0))
            slack = max(gap - self._r157_min_gap(v_obs), 0.0)
            v_ref = min(speed_ref, float(np.sqrt(v_obs ** 2 + 2.0 * decel * slack)))
        if v_ref <= 0.0:
            return 0.0
        if obstacle is not None and obstacle[0] <= self._r157_min_gap(ego_speed):
            return 0.0
        tol = float(self.rw.get("slow_tolerance", 0.0))
        shortfall = 1.0 - ego_speed / v_ref
        if shortfall <= tol:
            return 0.0
        return weight * (shortfall - tol) / (1.0 - tol)

    def _blocked_penalty(self, leader, speed_ref: float) -> float:
        """더 빠른 인접 차로를 두고 현재 차로에 머무는 기회비용만큼의 감점을 반환한다.

        감점 = ``blocked_penalty`` × max(0, 인접 차로 흐름 속도 − 현재 차로 흐름 속도)
        / ``speed_ref``. 차로 흐름 속도는 시야 안 앞차 속도이고, 앞차가 없으면
        ``speed_ref``다. 둘 다 ``speed_ref``를 넘지 않는다. 인접 차로는
        ``getBestLanes`` 기준으로 필요한 차선 변경 수가 0인 경로상 최적 차로만 비교한다.
        진출을 앞두고 다시 돌아와야 하는 차로는 비교하지 않으므로, 진출 대기 줄을
        벗어나도 이 감점은 줄지 않는다. ``leader``는 :meth:`_neighbor`가 반환한
        같은 차로 앞차의 (간격, 속도) 또는 None이다.
        """
        weight = float(self.rw.get("blocked_penalty", 0.0))
        edge = traci.vehicle.getRoadID(self.ego_id)
        if weight <= 0.0 or edge.startswith(":"):
            return 0.0
        current = speed_ref if leader is None else min(speed_ref, leader[1])
        best = current
        index = traci.vehicle.getLaneIndex(self.ego_id)
        best_lane = {lane[0]: lane[3] == 0 for lane in traci.vehicle.getBestLanes(self.ego_id)}
        for offset in (+1, -1):
            if not best_lane.get(f"{edge}_{index + offset}", False):
                continue
            adj_leader = self._neighbor(offset, leader=True)
            best = max(best, speed_ref if adj_leader is None
                       else min(speed_ref, adj_leader[1]))
        return weight * (best - current) / speed_ref

    def _speed_limit(self) -> float:
        """ego가 있는 차선의 제한속도. OSM 도로망이 아니면 vmax 고정."""
        if self.osm is None:
            return self.vmax
        try:
            lane = traci.vehicle.getLaneID(self.ego_id)
            return min(self.vmax, traci.lane.getMaxSpeed(lane))
        except traci.TraCIException:
            return self.vmax

    def get_detected_vehicles(self):
        """현재 관측 시야에 들어온(=감지된) 배경차들의 좌표 목록을 반환.

        데이터셋 수집용 부가 정보 — 관측 벡터에는 들어가지 않는다.
        "감지"의 정의는 관측과 동일: 관측 대상 차선(obs_offsets 범위)에서
        ego 기준 종방향 ±visibility(W) 이내에 있는 차량.

        Returns: list[dict]  각 원소 =
            {"id": 차량 id, "x": 절대 x, "y": 절대 y,
             "rel_pos": ego 기준 종방향 상대거리(m, 앞 +), "speed": m/s,
             "lane_offset": ego 차선 기준 오프셋(+왼쪽)}
        ego가 없으면 빈 리스트.
        """
        out = []
        if self.ego_id not in traci.vehicle.getIDList():
            return out
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            if edge.startswith(":"):
                return out
            ego_lane = traci.vehicle.getLaneIndex(self.ego_id)
            ego_pos = traci.vehicle.getLanePosition(self.ego_id)
            n_lanes = traci.edge.getLaneNumber(edge)
            for offset in self.obs_offsets:
                li = ego_lane + offset
                if li < 0 or li >= n_lanes:
                    continue
                for vid in traci.lane.getLastStepVehicleIDs(f"{edge}_{li}"):
                    if vid == self.ego_id:
                        continue
                    rel = traci.vehicle.getLanePosition(vid) - ego_pos
                    if abs(rel) > self.visibility:
                        continue
                    x, y = traci.vehicle.getPosition(vid)
                    out.append({"id": vid, "x": float(x), "y": float(y),
                                "rel_pos": float(rel),
                                "speed": float(traci.vehicle.getSpeed(vid)),
                                "lane_offset": int(offset)})
        except traci.TraCIException:
            pass
        return out

    def _get_obs(self):
        # ego가 이미 사라졌다면(충돌/도착) 0벡터 반환 — 종료 직전 마지막 관측용
        if self.ego_id not in traci.vehicle.getIDList():
            return np.zeros(self.obs_dim, dtype=np.float32)

        v = traci.vehicle.getSpeed(self.ego_id)
        obs = [v / self.vmax]                      # [0] 내 절대속도

        # ----- 5개 차선(좌2→좌1→현재→우1→우2)의 선행/후행 상대거리·상대속도 -----
        for offset in self.obs_offsets:
            lead = self._neighbor(offset, leader=True)
            if lead is None:
                obs += [1.0, 1.0]                  # 없음: 멀고 + 위협 없음
            else:
                gap, lv = lead
                obs += [gap / self.visibility,
                        float(np.clip((lv - v) / self.vmax, -1.0, 1.0))]

            foll = self._neighbor(offset, leader=False)
            if foll is None:
                obs += [-1.0, -1.0]                # 없음: 멀고 + 위협 없음
            else:
                gap, fv = foll
                obs += [-gap / self.visibility,    # 후행은 음수 부호
                        float(np.clip((fv - v) / self.vmax, -1.0, 1.0))]

        # ----- 5개 차선의 연결성 (같은 순서) -----
        for offset in self.obs_offsets:
            obs.append(self._lane_connectivity(offset))

        # ----- 5개 차선의 전방 차량 밀도 (같은 순서) -----
        for offset in self.obs_offsets:
            obs.append(self._lane_density(offset))

        # ----- 추가 컬럼 [31..44] -----
        obs += self._extra_obs()

        return np.clip(np.array(obs, dtype=np.float32), -1.0, 1.0)

    # ==================================================================
    # 1 스텝 진행: 행동 적용 → 시뮬레이션 1스텝 → 보상/종료 판정
    # ==================================================================
    def _scan_lane_neighbors(self, direction: int):
        """목표 차선을 직접 스캔해 (가장 가까운 앞차 gap·속도, 뒷차 gap·속도) 반환.

        getNeighbors의 백업. getNeighbors는 SUMO 차선변경모델의 이웃 캐시를
        읽는데, "방금 스폰된 차량"은 다음 스텝까지 이 캐시에 없어서
        도로 입구 근처에서 사각지대가 생긴다. 차선의 차량 목록을 직접
        훑으면 이미 삽입된 차량은 전부 잡힌다.
        """
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            li = traci.vehicle.getLaneIndex(self.ego_id) + direction
            # 존재하지 않는 차선이면 조회하지 않는다.
            # (없는 lane id를 traci.lane.*로 조회하면 SUMO가 매번
            #  에러 로그를 찍으므로, 범위 검사를 먼저 해서 조용히 건너뛴다)
            if edge.startswith(":") or li < 0 \
                    or li >= traci.edge.getLaneNumber(edge):
                return None, None
            lane_id = f"{edge}_{li}"
            my_pos = traci.vehicle.getLanePosition(self.ego_id)
            front, rear = None, None
            for vid in traci.lane.getLastStepVehicleIDs(lane_id):
                d = traci.vehicle.getLanePosition(vid) - my_pos
                if d >= 0:
                    g = d - traci.vehicle.getLength(vid)
                    if front is None or g < front[0]:
                        front = (g, traci.vehicle.getSpeed(vid))
                else:
                    g = -d - traci.vehicle.getLength(self.ego_id)
                    if rear is None or g < rear[0]:
                        rear = (g, traci.vehicle.getSpeed(vid))
            return front, rear
        except traci.TraCIException:
            return None, None

    def _overlap(self) -> float:
        """ego 차체가 현재 차선 경계를 넘어 걸친 폭(m)을 반환한다."""
        lane_w = traci.lane.getWidth(traci.vehicle.getLaneID(self.ego_id))
        car_w = traci.vehicle.getWidth(self.ego_id)
        return max(0.0, abs(traci.vehicle.getLateralLanePosition(self.ego_id))
                   + car_w / 2.0 - lane_w / 2.0)

    def _quantize_lane_change(self, raw: float) -> int:
        """차선변경 트리거 ``raw``를 -1(오른쪽), 0(유지), +1(왼쪽)로 변환한다."""
        if raw > self.lane_change_threshold:
            return 1
        if raw < -self.lane_change_threshold:
            return -1
        return 0

    def _lc_cooling_down(self) -> bool:
        """걸쳤다가 취소한 직후라 새 트리거를 받지 않는 중인지 반환한다."""
        return self.step_count < self.lc_block_until

    def _start_maneuver(self, command: int) -> bool:
        """``command`` 방향으로 차선변경 기동을 시작한다.

        기동이 없고 교차로 밖일 때만 시작한다. 목표 차선이 없으면 기동을 시작하지
        않고 False를 반환하며, 그 밖의 경우는 True를 반환한다.
        """
        if command == 0 or self.lc_dir != 0 or self._lc_cooling_down():
            return True
        edge = traci.vehicle.getRoadID(self.ego_id)
        if edge.startswith(":"):
            return True
        index = traci.vehicle.getLaneIndex(self.ego_id)
        target = index + command
        if target < 0 or target >= traci.edge.getLaneNumber(edge):
            return False
        self.lc_dir = command
        self.lc_target = target
        self.lc_edge = edge
        self.lc_straddled = False
        return True

    def _apply_lateral(self, lat_raw: float) -> None:
        """이번 스텝의 횡속도를 정하고 ego를 옆으로 움직인다.

        기동 중에는 ``lat_raw``를 기동 방향 기준 횡가속도로 적분한다. 차체가 처음
        차선 경계에 걸치기 전까지는 ``lat_raw`` 대신 목표 차선 쪽 최대 횡가속도를
        쓴다. 기동이 없으면 ``lat_raw``를 무시하고 현재 차선 중앙으로 정렬한다.
        교차로 내부에서는 SUMO가 연결 차선을 따라가므로 횡이동을 하지 않는다.
        """
        if traci.vehicle.getRoadID(self.ego_id).startswith(":"):
            self.lat_speed = 0.0
            return
        dt = self.step_length
        dv_max = self.max_lat_accel * dt
        if self.lc_dir != 0:
            accel = lat_raw if self.lc_straddled else 1.0
            self.lat_speed += self.lc_dir * accel * dv_max
        else:
            # 남은 거리 안에 멈출 수 있는 속도로 차선 중앙에 다가간다.
            lat = traci.vehicle.getLateralLanePosition(self.ego_id)
            target_speed = -np.sign(lat) * min(
                self.max_speed_lat,
                np.sqrt(2.0 * self.max_lat_accel * abs(lat)),
                abs(lat) / dt)
            self.lat_speed += float(np.clip(target_speed - self.lat_speed,
                                            -dv_max, dv_max))
        # 횡속도 상한은 종방향 속도에 비례한다 (정지 시 0)
        lat_cap = min(self.max_speed_lat,
                      self.lat_speed_per_speed * traci.vehicle.getSpeed(self.ego_id))
        self.lat_speed = float(np.clip(self.lat_speed, -lat_cap, lat_cap))

        # SUMO 자체 횡속도 한계가 이 명령을 더 줄이지 않도록 매 스텝 맞춘다
        traci.vehicle.setMaxSpeedLat(self.ego_id, max(abs(self.lat_speed), 1e-3))
        traci.vehicle.changeSublane(self.ego_id, self.lat_speed * dt)

    def _update_maneuver(self) -> tuple[int, bool]:
        """시뮬레이션 스텝 뒤 기동의 걸침, 완료, 취소를 판정한다.

        차체가 한 차선 안에 완전히 들어오면 기동이 끝난다. 목표 차선이면 완료,
        출발 차선이면 취소다. 걸친 채 교차로에 들어가거나 기동 중 도로가 바뀌면
        취소로 끝낸다.

        Returns ``(완료된 차선변경 방향, 걸친 채 교차로에 들어갔는지)``이다.
        완료되지 않았으면 방향은 0이다.
        """
        if self.lc_dir == 0:
            return 0, False
        overlap = self._overlap()
        edge = traci.vehicle.getRoadID(self.ego_id)
        if edge != self.lc_edge:
            junction_straddle = edge.startswith(":") and overlap > 0.0
            if self.lc_straddled:
                self.lc_block_until = self.step_count + self.lc_cooldown_steps
            self._reset_maneuver()
            return 0, junction_straddle
        if overlap > 0.0:
            if self.lc_straddled:
                self.lc_straddle_steps += 1
            self.lc_straddled = True
            return 0, False
        if not self.lc_straddled:
            return 0, False
        completed = self.lc_dir if (
            traci.vehicle.getLaneIndex(self.ego_id) == self.lc_target) else 0
        if completed == 0:
            # 걸쳤다가 출발 차선으로 돌아온 취소: 잠시 다시 트리거하지 못하게 한다
            self.lc_block_until = self.step_count + self.lc_cooldown_steps
        self._reset_maneuver()
        return completed, False

    def get_privileged_state(self) -> dict:
        """POMDP 데이터셋용 simulator-side state.

        agent가 실제로 받는 입력은 _get_obs()의 부분관측이고, 이 메서드는
        분석/지도신호용 privileged state만 수집한다. 정책 입력으로 쓰지 않는다.
        """
        if self.ego_id not in traci.vehicle.getIDList():
            return {"ego_alive": False}
        try:
            # 입체 도로에서는 xy가 겹쳐도 다른 층일 수 있어 z까지 기록한다
            x, y, z = traci.vehicle.getPosition3D(self.ego_id)
            vehicles = []
            for vid in traci.vehicle.getIDList():
                try:
                    vx, vy, vz = traci.vehicle.getPosition3D(vid)
                    vehicles.append({
                        "id": vid,
                        "x": float(vx),
                        "y": float(vy),
                        "z": float(vz),
                        "speed": float(traci.vehicle.getSpeed(vid)),
                        "lane": int(traci.vehicle.getLaneIndex(vid)),
                        "lane_id": traci.vehicle.getLaneID(vid),
                        "road_id": traci.vehicle.getRoadID(vid),
                        "lane_pos": float(traci.vehicle.getLanePosition(vid)),
                    })
                except traci.TraCIException:
                    continue
            return {
                "ego_alive": True,
                "ego_x": float(x),
                "ego_y": float(y),
                "ego_z": float(z),
                "ego_speed": float(traci.vehicle.getSpeed(self.ego_id)),
                "ego_lane": int(traci.vehicle.getLaneIndex(self.ego_id)),
                "ego_lane_id": traci.vehicle.getLaneID(self.ego_id),
                "ego_road_id": traci.vehicle.getRoadID(self.ego_id),
                "ego_lane_pos": float(traci.vehicle.getLanePosition(self.ego_id)),
                "vehicles": vehicles,
            }
        except traci.TraCIException:
            return {"ego_alive": False}

    def step(self, action):
        # action = [가감속 raw, 차선변경 트리거 raw, 기동 횡가속도 raw]
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.size < self.act_dim:
            raise ValueError(
                f"action은 {self.act_dim}차원이어야 합니다. 받은 shape={np.asarray(action).shape}")

        accel_raw = float(np.clip(action_arr[self.accel_index], -1.0, 1.0))
        lc_raw = float(np.clip(action_arr[self.lane_change_index], -1.0, 1.0))
        lat_raw = float(np.clip(action_arr[self.lat_accel_index], -1.0, 1.0))
        lc_command = self._quantize_lane_change(lc_raw)
        lateral_valid, lc_started, lc_used, lat_used = True, False, False, False
        lane_before, light_before, odometer_before = None, None, 0.0

        # ----- (1) 종방향 + 횡방향 행동 적용 -----
        if self.ego_id in traci.vehicle.getIDList():
            v = traci.vehicle.getSpeed(self.ego_id)
            lane_before = int(traci.vehicle.getLaneIndex(self.ego_id))
            light_before = self._next_light()
            odometer_before = traci.vehicle.getDistance(self.ego_id)

            accel_cmd = (accel_raw * self.max_accel
                         if accel_raw >= 0.0 else accel_raw * self.max_decel)
            dv = accel_cmd * self.step_length
            # 제한속도가 낮은 구간에 들어서면 최대 감속도까지만 줄인다
            limit = max(self._speed_limit(), v - self.max_decel * self.step_length)
            traci.vehicle.setSpeed(
                self.ego_id, float(np.clip(v + dv, 0.0, limit)))

            # 이번 스텝에 [1]과 [2]가 실제로 쓰였는지 기록한다.
            in_junction = traci.vehicle.getRoadID(self.ego_id).startswith(":")
            lc_used = self.lc_dir == 0 and not in_junction and not self._lc_cooling_down()
            lat_used = self.lc_dir != 0 and self.lc_straddled and not in_junction
            lateral_valid = self._start_maneuver(lc_command)
            lc_started = lc_used and self.lc_dir != 0
            self._apply_lateral(lat_raw)

        # ----- (2) 시뮬레이션 1스텝 -----
        traci.simulationStep()
        self.step_count += 1
        self.total_steps += 1

        # ----- (3) 이벤트/관측 -----
        collided = self.ego_id in traci.simulation.getCollidingVehiclesIDList()
        arrived = self.ego_id in traci.simulation.getArrivedIDList()
        alive = self.ego_id in traci.vehicle.getIDList()
        lane_after = int(traci.vehicle.getLaneIndex(self.ego_id)) if alive else None
        # 기동이 목표 차선에서 끝난 경우만 차선변경으로 센다.
        # 걸쳤다가 출발 차선으로 돌아오면 차선 인덱스가 잠시 바뀌어도 세지 않는다.
        lane_change, junction_straddle = 0, False
        exit_missed, queue_cut = False, 0
        if alive:
            lane_change, junction_straddle = self._update_maneuver()
            exit_missed = self._exit_missed()
            if lane_change != 0:
                queue_cut = self._queue_cut_count()

        # 신호 위반: 스텝 시작 때 적색이던 신호의 정지선을 이번 스텝에 넘었는가.
        #   (이번 스텝 주행거리 ≥ 스텝 시작 때 정지선까지 거리)
        red_light = False
        if (alive and light_before is not None
                and _SIGNAL_VALUE.get(light_before[3], 0.0) < 0):
            moved = traci.vehicle.getDistance(self.ego_id) - odometer_before
            red_light = moved >= light_before[2]

        obs = self._get_obs()
        terminated, truncated = False, False
        ego_speed, ego_gap = None, None
        nav_shaping = 0.0

        # ----- (4) 기존 보상 유지 -----
        if collided:
            reward = -float(self.rw["collision_penalty"])
            terminated = True
        elif red_light:
            # 적색 신호 통과: 충돌과 같은 크기의 벌점 후 종료
            reward = -float(self.rw.get("red_light_penalty", 0.0))
            terminated = True
        elif exit_missed:
            # 출구 놓침: 경로로 이어지지 않는 차선 끝에 도달하면 벌점 후 종료
            reward = -float(self.rw.get("exit_miss_penalty", 0.0))
            terminated = True
        elif arrived:
            reward = float(self.rw["arrival_bonus"])
            terminated = True
        elif not alive:
            reward = 0.0
            terminated = True
        else:
            ego_speed = traci.vehicle.getSpeed(self.ego_id)
            same_lane_leader = self._neighbor(0, leader=True)
            if same_lane_leader is not None:
                ego_gap = same_lane_leader[0]
            # 속도 보상은 현재 구간 제한속도 기준 (OSM이 아니면 vmax와 동일)
            speed_ref = self._speed_limit()
            # 진행 거리 보상: 이번 스텝 이동거리를 제한속도로 달렸을 때의 거리로 나눈다.
            # 경로 전체의 보상 합이 경로 길이로 정해지므로 일찍 도착할수록 유리하다.
            moved = traci.vehicle.getDistance(self.ego_id) - odometer_before
            reward = float(self.rw["speed_weight"]) * min(
                moved / (speed_ref * self.step_length), 1.0)
            # ---- 안전거리 벌점 (UN R157 속도별 최소 차간거리) ----
            # getLeader의 gap은 내 minGap을 뺀 값이라 다시 더해 실제 간격으로 쓴다.
            # 위반 정도(1 - 간격/d_min)의 제곱에 비례해 감점한다. d_min 근처의 작은 위반은 약하게,
            # 크게 붙을수록 강하게 감점한다.
            if ego_gap is not None:
                gap = ego_gap + traci.vehicle.getMinGap(self.ego_id)
                d_min = self._r157_min_gap(ego_speed)
                if gap < d_min:
                    reward -= (float(self.rw.get("safe_gap_penalty", 0.0))
                               * (1.0 - max(gap, 0.0) / d_min) ** 2)

            reward -= self._slow_penalty(ego_speed, same_lane_leader, speed_ref)

            # ---- 겹침 벌점: 차체가 차선 경계를 넘어 걸친 폭에 비례 ----
            # 기동 중에는 처음 걸친 뒤 overlap_grace 초 동안 감점하지 않는다. 정상적인
            # 차선변경은 이 안에 끝나므로 감점 없이, 걸친 채 오래 버티는 경우만 감점한다.
            # 기동 밖에서 걸쳐 있는 경우(취소된 채 교차로를 지난 뒤 등)는 바로 감점한다.
            car_w = traci.vehicle.getWidth(self.ego_id)
            grace = float(self.rw.get("overlap_grace", 0.0))
            if self.lc_dir == 0 or self.lc_straddle_steps * self.step_length >= grace:
                reward -= (float(self.rw.get("overlap_penalty", 0.0))
                           * self._overlap() / car_w)

            # ---- 걸친 채 교차로 진입 벌점 (기동은 취소로 끝난다) ----
            if junction_straddle:
                reward -= float(self.rw.get("junction_overlap_penalty", 0.0))

            # ---- 막힘 페널티: 더 빠른 인접 차로를 두고 머무는 기회비용만큼 매 스텝 감점 ----
            # "차선을 유지한 채 막혀 있기"와 "변경으로 탈출하기" 사이에
            # 즉각적인 보상 차등을 만들어 추월 학습 신호를 조밀하게 한다.
            # (자세한 설계 이유는 env/mdp_config.py의 REWARD 주석 참고)
            reward -= self._blocked_penalty(same_lane_leader, speed_ref)

            # ---- 차선이 없는 방향 차선변경 트리거 감점 ----
            # → 의미 없는 행동 출력을 줄이는 정칙화 (rl_action_penalty 대응)
            if not lateral_valid:
                reward -= float(self.rw.get("invalid_action_penalty", 0.0))

            # ---- 대기 줄 앞지르기: 합류 완료 시 밀려난 대기 줄 차량 수에 비례 (상한 적용) ----
            queue_cut_cost = float(self.rw.get("queue_cut_penalty", 0.0)) * queue_cut
            queue_cut_cap = float(self.rw.get("queue_cut_max_penalty", 0.0))
            if queue_cut_cap > 0.0:
                queue_cut_cost = min(queue_cut_cost, queue_cut_cap)
            reward -= queue_cut_cost

            # ---- 내비게이션 포텐셜 셰이핑: 출구 접근 시 최적 차선 이동에 즉각 보상 ----
            # 교차로 내부는 _best_lane_info가 중립값(0, route_horizon)을 반환하므로 직전 Φ 유지.
            if not traci.vehicle.getRoadID(self.ego_id).startswith(":"):
                new_phi = self._nav_potential()
                nav_shaping = new_phi - self._nav_phi
                self._nav_phi = new_phi
                reward += nav_shaping

        if self.step_count >= self.episode_max_steps:
            truncated = True

        info = {
            "collided": collided,
            "red_light": red_light,
            "arrived": arrived,
            "speed": ego_speed,
            "gap": ego_gap,
            "accel_raw": accel_raw,
            "lane_change_raw": lc_raw,
            "lat_accel_raw": lat_raw,
            "lane_change_used": lc_used,
            "lat_accel_used": lat_used,
            "lane_change_started": lc_started,
            "lane_change": lane_change,
            "lane_change_applied": lane_change != 0,
            "lane_change_dir": self.lc_dir,
            "straddle_time": self.lc_straddle_steps * self.step_length,
            "npc_emergency_decel": self.npc_emergency_decel,
            "junction_straddle": junction_straddle,
            "exit_missed": exit_missed,
            "queue_cut": queue_cut,
            "nav_shaping": nav_shaping,
            "lane_before": lane_before,
            "lane_after": lane_after,
        }
        return obs, float(reward), terminated, truncated, info

    # ==================================================================
    def close(self):
        """환경 정리. 학습이 끝나면 반드시 호출해서 SUMO 프로세스를 종료할 것."""
        if self._started:
            traci.close()
            self._started = False
