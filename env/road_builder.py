"""configs/road_config.py 를 읽어 SUMO 도로 파일들을 자동 생성하는 모듈.

SUMO는 도로를 여러 XML 파일로 정의한다:
    highway.nod.xml  : 노드(교차점/도로 끝점)의 좌표
    highway.edg.xml  : 엣지(도로 구간) — 어느 노드에서 어느 노드로, 차선 수, 제한속도
    highway.con.xml  : 차선 간 연결(connection) — 사거리의 차로 확장/회전 포켓 연결
    highway.net.xml  : 위 두 파일을 netconvert 도구로 컴파일한 "완성된 도로망"
                       (SUMO가 실제로 읽는 파일. 직접 작성하기엔 복잡해서 도구로 생성)
    highway.rou.xml  : 차량 타입(vType)과 교통 수요(flow/route) 정의
    highway.sumocfg  : 위 파일들을 묶어주는 설정 파일 (시뮬레이션 실행의 진입점)

train.py / test.py 가 시작할 때마다 build()를 호출하므로,
road_config.py만 수정하면 도로가 자동으로 다시 만들어진다.
"""
import os
import subprocess

import sumolib
from sumolib import checkBinary  # SUMO 실행파일(sumo, netconvert 등) 경로를 찾아주는 유틸
from env import osm_road
from env.mdp_config import SIMULATION
try:
    from env.road_config import SCENERY
except ImportError:      # 구버전 설정 하위호환
    SCENERY = {"trees": False}


def build(road: dict, ego: dict, traffic: dict, out_dir: str) -> str:
    """SUMO 파일들을 out_dir에 생성하고, sumocfg 파일 경로를 반환한다.

    Args:
        road:    configs.road_config.ROAD    (도로 형상)
        ego:     configs.road_config.EGO     (RL 차량 속성)
        traffic: configs.road_config.TRAFFIC (배경 교통류)
        out_dir: 생성 위치 (보통 프로젝트의 sumo/ 폴더)
    """
    os.makedirs(out_dir, exist_ok=True)
    layout = intersection_layout(road)
    osm = osm_config(road)

    if osm is not None:
        # ──────────────────────────────────────────────────────
        # 1~2) OSM 실측 입체 도로망: netconvert까지 여기서 끝낸다
        #      (자세한 구조는 env/osm_road.py 주석 참고)
        # ──────────────────────────────────────────────────────
        net_path = osm_road.write_network(out_dir, osm)
        # 램프로 빠지려면 전략적 차선변경이 필요하므로 keep_lane을 끈다
        traffic = dict(traffic, controllers={
            k: dict(c, keep_lane=False)
            for k, c in (traffic.get("controllers") or {}).items()})
    elif layout is None:
        # ──────────────────────────────────────────────────────
        # 1) 노드 파일: 직선 도로의 양 끝점 2개
        #    n0 = 원점(출발), n1 = length만큼 떨어진 지점(도착)
        # ──────────────────────────────────────────────────────
        with open(os.path.join(out_dir, "highway.nod.xml"), "w") as f:
            f.write(f"""<nodes>
    <node id="n0" x="0" y="0"/>
    <node id="n1" x="{road['length']}" y="0"/>
</nodes>
""")

        # ──────────────────────────────────────────────────────
        # 2) 엣지 파일: n0 → n1 을 잇는 도로 구간 하나
        #    numLanes = 차선 수, speed = 제한속도(m/s)
        # ──────────────────────────────────────────────────────
        with open(os.path.join(out_dir, "highway.edg.xml"), "w") as f:
            f.write(f"""<edges>
    <edge id="e0" from="n0" to="n1" numLanes="{road['num_lanes']}" speed="{road['speed_limit']}"/>
</edges>
""")
        # 직선 도로는 연결 정보를 netconvert가 알아서 만든다 (빈 파일)
        with open(os.path.join(out_dir, "highway.con.xml"), "w") as f:
            f.write("<connections/>\n")
    else:
        # ──────────────────────────────────────────────────────
        # 1~2) 사거리: 노드 / 엣지 / 차선 연결(connection) 파일
        #      (자세한 구조는 _write_intersection_network 주석 참고)
        # ──────────────────────────────────────────────────────
        _write_intersection_network(out_dir, layout)

    # ──────────────────────────────────────────────────────
    # 3) 라우트 파일: 차량 타입 2종 + 배경 교통 flow
    #
    #    <vType>  : 차량의 물리 속성 템플릿
    #       ego_type : RL이 조종할 차 (빨간색, color="1,0,0" = RGB)
    #    <vTypeDistribution> : 배경차 컨트롤러 믹스.
    #       TRAFFIC["controllers"]에 정의된 여러 차량추종모델(Krauss/IDM/
    #       EIDM/ACC 등)을 probability 비율로 섞는다. flow가 이 분포를
    #       type으로 쓰면, 투입되는 차량마다 SUMO가 확률적으로 타입을 뽑는다.
    #    <route>  : 지나갈 엣지들의 나열. 여기선 e0 하나뿐.
    #    <flow>   : 교통 수요 정의. period="X"는 "X초마다 정확히 1대씩"
    #               → 등간격(uniform) 생성. (vehsPerHour와 등가지만
    #               "비슷한 간격으로 새로 생성"을 명시적으로 보장)
    #               ego는 여기에 없고, 환경(sumo_env.py)이 reset마다
    #               traci.vehicle.add()로 직접 투입한다.
    # ──────────────────────────────────────────────────────
    lc_model = road.get("lane_change_model", "LC2013")
    controllers = traffic.get("controllers")
    if controllers:
        # probability 합이 1이 아니어도 되도록 자동 정규화
        total_p = sum(float(c.get("probability", 1.0))
                      for c in controllers.values())
        vtype_lines = []
        gaps = npc_min_gaps(traffic)
        for name, c in controllers.items():
            prob = float(c.get("probability", 1.0)) / total_p
            attrs = [
                f'carFollowModel="{c["carFollowModel"]}"',
                f'laneChangeModel="{lc_model}"',
                f'accel="{c.get("accel", 2.5)}"',
                f'decel="{c.get("decel", 4.5)}"',
                f'tau="{c.get("tau", 1.0)}"',
                # 비상 제동 한계와 반응 주기. 컨트롤러별 값이 없으면 공통값을 쓴다.
                f'emergencyDecel="{c.get("emergency_decel", traffic.get("emergency_decel", 9.0))}"',
                f'actionStepLength="{c.get("action_step_length", traffic.get("action_step_length", 0.0))}"',
                # 컨트롤러별 희망속도 오버라이드 (없으면 공통 max_speed).
                # 컨트롤러마다 속도를 다르게 주면 차선 간 속도차가 생겨
                # "추월할 이유"가 자연스럽게 만들어진다.
                f'maxSpeed="{c.get("max_speed", traffic["max_speed"])}"',
                # speedDev: 같은 타입 안에서도 개체별 희망속도 편차 (0~)
                f'speedDev="{c.get("speed_dev", 0.1)}"',
                f'color="{c.get("color", "1,1,0")}"',
                'vClass="passenger"', 'guiShape="passenger/sedan"',
            ]
            # sigma는 Krauss 계열에만 의미가 있으므로 지정된 경우에만 넣는다
            if "sigma" in c:
                attrs.append(f'sigma="{c["sigma"]}"')
            # ── 배경차 차선 유지 (keep_lane, 기본 True) ──
            # 배경차의 자율 차선변경(전략/협조/속도이득/우측유지)을 전부 꺼서
            # 각자 자기 차선을 지키게 한다. 이유:
            #   배경차가 자유롭게 돌아다니면 (측정: 150스텝에 477회 변경)
            #   차선 간 틈이 수시로 닫혀서, ego가 틈을 노려 파고드는
            #   "칼치기 추월"을 학습할 안정적인 기회 자체가 사라진다.
            #   배경차가 차선을 유지하면 "느린 차 뒤 → 옆 틈으로 변경 →
            #   가속"이라는 추월 패턴이 재현 가능한 형태로 나타난다.
            # 특정 컨트롤러만 돌아다니게 하려면 그 항목에 "keep_lane": False.
            if c.get("keep_lane", True):
                attrs += ['lcStrategic="0"', 'lcCooperative="0"',
                          'lcSpeedGain="0"', 'lcKeepRight="0"']
                if lc_model == "SL2015":
                    attrs.append('lcSublane="0"')
            # minGap 단계마다 같은 컨트롤러의 vType을 하나씩 만들고 확률을 나눈다.
            for vid, gap in zip(npc_vtype_ids(name, traffic), gaps or [c.get("min_gap", 2.5)]):
                vtype_lines.append(
                    f'        <vType id="{vid}" probability="{prob / len(gaps or [0]):.4f}" '
                    f'minGap="{gap:.3f}" ' + " ".join(attrs) + "/>")
        vtypes_block = ('    <vTypeDistribution id="carDist">\n'
                        + "\n".join(vtype_lines)
                        + "\n    </vTypeDistribution>")
        flow_type = "carDist"
    else:
        # controllers 설정이 없으면 예전 방식(단일 Krauss)으로 동작 (하위호환)
        vtypes_block = (f'    <vType id="car" accel="2.5" decel="4.5" '
                        f'laneChangeModel="{lc_model}" '
                        f'maxSpeed="{traffic["max_speed"]}" '
                        f'sigma="{traffic.get("sigma", 0.5)}" '
                        f'emergencyDecel="{traffic.get("emergency_decel", 9.0)}" '
                        f'actionStepLength="{traffic.get("action_step_length", 0.0)}" '
                        f'vClass="passenger" guiShape="passenger/sedan"/>')
        flow_type = "car"

    if osm is not None:
        demand_block = osm_road.demand_block(net_path, osm, flow_type, traffic)
    elif layout is None:
        demand_block = f"""    <route id="r0" edges="e0"/>
    <flow id="traffic" type="{flow_type}" route="r0" begin="0" end="100000"
          period="{3600.0 / traffic['vehs_per_hour']:.2f}"
          departLane="{traffic['depart_lane']}" departSpeed="{traffic['depart_speed']}"/>"""
    else:
        demand_block = _intersection_demand(layout, traffic, flow_type)

    with open(os.path.join(out_dir, "highway.rou.xml"), "w") as f:
        f.write(f"""<routes>
    <vType id="ego_type" accel="{ego['accel']}" decel="{ego['decel']}" maxSpeed="{ego['max_speed']}" laneChangeModel="{lc_model}" maxSpeedLat="{ego.get('max_speed_lat', 1.0)}" lcAccelLat="{ego.get('lc_accel_lat', 1.0)}" color="1,0,0" vClass="passenger" guiShape="passenger/sedan"/>
{vtypes_block}
{demand_block}
</routes>
""")

    # ──────────────────────────────────────────────────────
    # 4) GUI 기본 시각화 설정
    #    GUI 관련 값의 source of truth는 env/mdp_config.py의 SIMULATION 하나뿐이다.
    #    view_road.py / train.py / test.py / 데이터 수집 모두 build()를 거치므로
    #    gui_zoom/gui_delay를 바꾸면 생성되는 highway.gui.xml에도 즉시 반영된다.
    # ──────────────────────────────────────────────────────
    # GUI 시작 화면: "도로 시작 구간"이 차선까지 보이게 자동 확대해둔다.
    #   (12km 전체 맞춤으로 시작하면 도로가 가느다란 선으로만 보여 직관성이
    #    떨어지므로, ego가 등장할 시작부를 확대해 대기 화면으로 삼는다.
    #    ▶ 재생 후에는 env가 ego를 추적(trackVehicle)하며 시점을 가져간다.)
    # 좌표/배율은 도로 길이·차선 수에서 "계산"하므로 설정을 바꿔도 안 어긋난다:
    #   zoom(%) = 도로길이 / 보고싶은폭(gui_view_width) × 100
    #     (SUMO zoom은 "네트워크 전체 = 100%" 기준)
    #   y 중심 = -차선폭(3.2m) × 차선수 / 2   (SUMO는 차선을 y<0 쪽으로 그림)
    gui_delay = float(SIMULATION.get("gui_delay", 100.0))
    view_width = float(SIMULATION.get("gui_view_width", 500.0))   # 시작 화면 시야(m)
    zoom = float(road["length"]) / view_width * 100.0
    center_x = view_width * 0.4                                   # 시작점이 화면 왼쪽에 오게
    center_y = -3.2 * int(road["num_lanes"]) / 2.0
    if layout is not None:
        # 사거리가 있으면 시작 화면을 교차로 중심에 맞춰 차로 확장 구간이 보이게 한다
        center_x, center_y = layout["position"], 0.0
    if osm is not None:
        # OSM 도로망은 좌표계가 달라 시작 화면을 도로망 중심에 맞춘다
        xmin, ymin, xmax, ymax = sumolib.net.readNet(net_path).getBoundary()
        zoom = max(xmax - xmin, ymax - ymin) / view_width * 100.0
        center_x, center_y = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    gui_settings_path = os.path.join(out_dir, "highway.gui.xml")
    with open(gui_settings_path, "w") as f:
        f.write(f"""<viewsettings>
    <scheme name="real world"/>
    <viewport zoom="{zoom:.0f}" x="{center_x:.1f}" y="{center_y:.1f}" angle="0"/>
    <delay value="{gui_delay}"/>
</viewsettings>
""")

    # ──────────────────────────────────────────────────────
    # 4.7) 도로변 장식(나무) — 순수 시각 요소
    #
    # SUMO의 <poly>(다각형)로 "초록 수관 + 갈색 줄기"를 그려 도로
    # 양옆에 심는다. additional 파일로 로드되며 차량 물리와 완전히
    # 무관하다 (충돌/관측/보상 어디에도 안 잡힘).
    # 좌표계: 차선은 y ∈ [0, -3.2×차선수] 영역 → 왼쪽 숲은 y > 0,
    # 오른쪽 숲은 y < -3.2×차선수 - side_offset 에 배치.
    # ──────────────────────────────────────────────────────
    # OSM 도로망은 도로가 곡선이라 직선 기준의 나무 배치를 쓸 수 없다 (빈 파일)
    if osm is None:
        _write_scenery(out_dir, road, layout)
    else:
        with open(os.path.join(out_dir, "highway.scenery.xml"), "w") as f:
            f.write("<additional/>\n")

    # ──────────────────────────────────────────────────────
    # 5) sumocfg: "이 net 파일과 이 route 파일로 시뮬레이션해라"
    # ──────────────────────────────────────────────────────
    lat_res = road.get("lateral_resolution")
    processing_block = ""
    if lat_res is not None:
        processing_block = f"""    <processing>
        <lateral-resolution value="{lat_res}"/>
    </processing>
"""
    cfg_path = os.path.join(out_dir, "highway.sumocfg")
    with open(cfg_path, "w") as f:
        f.write(f"""<configuration>
    <input>
        <net-file value="highway.net.xml"/>
        <route-files value="highway.rou.xml"/>
        <additional-files value="highway.scenery.xml"/>
    </input>
{processing_block}    <gui_only>
        <gui-settings-file value="highway.gui.xml"/>
    </gui_only>
</configuration>
""")

    # ──────────────────────────────────────────────────────
    # 6) netconvert 실행: nod + edg → net.xml 컴파일
    #    (SUMO 설치 시 함께 깔리는 커맨드라인 도구)
    # ──────────────────────────────────────────────────────
    if osm is not None:
        return cfg_path
    netconvert = checkBinary("netconvert")
    subprocess.run(
        [netconvert,
         "--node-files", os.path.join(out_dir, "highway.nod.xml"),
         "--edge-files", os.path.join(out_dir, "highway.edg.xml"),
         "--connection-files", os.path.join(out_dir, "highway.con.xml"),
         "--no-turnarounds", "true",   # 도로 끝/교차로에서 U턴 연결 생성 안 함
         # 노드 좌표를 그대로 유지 (나무 배치/GUI 시점이 같은 좌표계를 쓰므로)
         "--offset.disable-normalization", "true",
         "-o", os.path.join(out_dir, "highway.net.xml")],
        check=True,           # 실패하면 예외를 던져서 바로 알 수 있게
        capture_output=True,  # 출력은 조용히 삼킴
    )
    return cfg_path


# ══════════════════════════════════════════════════════════════
# 사거리(4지 교차로) 도로망
#
#                    N0
#                    │   (교차 도로, cross_length)
#                    NF
#   W0 ── W_in0 ── WF ┼ C ┼ EF ── E_in0 ── E0
#        (ego 출발)   SF        (ego 도착 = 주도로 끝)
#                    │
#                    S0
#
# - 주도로(W↔E)는 x축, 교차도로(N↔S)는 x=position 위치의 y축.
# - 모든 도로는 양방향. SUMO는 차선을 진행방향 "오른쪽"에 그리므로
#   동쪽행(ego 방향)은 y<0, 서쪽행은 y>0 쪽에 놓인다.
# - 교차로에 가까워지면 접근로의 차로 수가 단계적으로 늘어난다(flare):
#     flare 한 단계 = (교차로까지 거리, 좌회전 포켓 수, 우회전 포켓 수)
#   해당 거리 지점에 노드를 두고, 그 뒤 엣지의 차로 수를
#   "기본 차로 + 좌포켓 + 우포켓"으로 늘린다.
# - 한 엣지 안의 차선 인덱스(0 = 맨 오른쪽):
#     [0 .. r-1]           우회전 포켓
#     [r .. r+n-1]         직진 차로 (기본 차로 n개)
#     [r+n .. r+n+l-1]     좌회전 포켓
#   새 포켓은 인접한 직진 차로에서 갈라져 나온다 (그 차로에서 두 방향 연결).
# ══════════════════════════════════════════════════════════════
LANE_WIDTH = 3.2   # SUMO 기본 차선 폭 (m)


def _norm_flare(flare, arm_length: float):
    """flare 설정을 (거리, 좌포켓, 우포켓) 리스트로 정리.

    먼 지점 → 교차로 순으로 정렬하고, 포켓 수는 교차로로 갈수록
    줄어들지 않게(누적 증가) 보정한다.
    """
    steps = sorted(((float(d), int(l), int(r)) for d, l, r in (flare or [])),
                   reverse=True)
    out, l_max, r_max = [], 0, 0
    for d, l, r in steps:
        if not (0.0 < d < arm_length - 10.0):
            raise ValueError(f"flare 거리 {d}m가 도로 구간 길이({arm_length}m) 범위를 벗어남")
        l_max, r_max = max(l_max, l), max(r_max, r)
        out.append((d, l_max, r_max))
    return out


def npc_min_gaps(traffic: dict) -> list[float]:
    """배경 차량 minGap(m) 단계 목록을 반환한다.

    ``traffic["min_gap_range"]``가 (하한, 상한)이면 그 사이를
    ``traffic["min_gap_levels"]``개로 고르게 나눈 값이다. 없으면 빈 목록이며,
    이때는 컨트롤러별 ``min_gap``을 쓴다.
    """
    rng = traffic.get("min_gap_range")
    if not rng:
        return []
    lo, hi = float(rng[0]), float(rng[1])
    n = max(int(traffic.get("min_gap_levels", 2)), 2)
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def npc_vtype_ids(name: str, traffic: dict) -> list[str]:
    """컨트롤러 ``name``의 vType id 목록을 반환한다. minGap 단계마다 하나씩이다."""
    gaps = npc_min_gaps(traffic)
    return [f"car_{name}_g{i}" for i in range(len(gaps))] if gaps else [f"car_{name}"]


def osm_config(road: dict):
    """ROAD["osm"]이 켜져 있으면 그 설정, 아니면 None."""
    osm = road.get("osm") or {}
    return osm if osm.get("enabled", False) else None


def intersection_layout(road: dict):
    """ROAD["intersection"] 설정을 노드/엣지 단위의 구조로 풀어낸다.

    사거리가 꺼져 있으면 None (= 기존 직선 도로).
    build(), 프리필(prefill_plan), 나무 배치가 모두 이 결과를 공유한다.
    """
    inter = road.get("intersection") or {}
    if osm_config(road) is not None or not inter.get("enabled", False):
        return None

    length = float(road["length"])
    pos = float(inter["position"])
    cross_len = float(inter.get("cross_length", 500.0))
    if not (0.0 < pos < length):
        raise ValueError("intersection.position은 0 < position < ROAD['length'] 이어야 함")

    main = {"lanes": int(road["num_lanes"]), "speed": float(road["speed_limit"]),
            "priority": 2, "flare": inter.get("main_flare", [])}
    cross = {"lanes": int(inter.get("cross_num_lanes", 2)),
             "speed": float(inter.get("cross_speed_limit", 13.89)),
             "priority": 1, "flare": inter.get("cross_flare", [])}
    # 팔(arm): 교차로 중심 C에서 바깥쪽을 가리키는 단위벡터 u와 길이
    arms_def = {
        "W": (main, (-1, 0), pos),
        "E": (main, (1, 0), length - pos),
        "N": (cross, (0, 1), cross_len),
        "S": (cross, (0, -1), cross_len),
    }
    nodes = {"C": (pos, 0.0)}
    arms = {}
    for a, (road_cls, u, arm_len) in arms_def.items():
        flare = _norm_flare(road_cls["flare"], arm_len)
        pt = lambda d: (pos + u[0] * d, u[1] * d)
        nodes[f"{a}0"] = pt(arm_len)
        # 접근로(교차로로 들어오는 방향) 구간: 바깥 끝 → flare 노드들 → C
        stops = [(arm_len, f"{a}0", 0, 0)] + [
            (d, f"{a}F{k}", l, r) for k, (d, l, r) in enumerate(flare)]
        segs = []
        for k, (d, nid, l, r) in enumerate(stops):
            if nid != f"{a}0":
                nodes[nid] = pt(d)
            d_next, to = (stops[k + 1][0], stops[k + 1][1]) if k + 1 < len(stops) else (0.0, "C")
            segs.append({"id": f"{a}_in{k}", "from": nid, "to": to,
                         "length": d - d_next, "left": l, "right": r,
                         "lanes": road_cls["lanes"] + l + r})
        arms[a] = {"u": u, "length": arm_len, "lanes": road_cls["lanes"],
                   "speed": road_cls["speed"], "priority": road_cls["priority"],
                   "segs": segs, "out": f"{a}_out",
                   "max_lanes": max(s["lanes"] for s in segs)}

    # 진행 방향 기준 직진/우회전/좌회전 목적지 팔
    by_u = {v["u"]: k for k, v in arms.items()}
    for a, arm in arms.items():
        hx, hy = -arm["u"][0], -arm["u"][1]          # 교차로로 들어가는 진행 방향
        arm["to"] = {"straight": by_u[(hx, hy)],
                     "right": by_u[(hy, -hx)],
                     "left": by_u[(-hy, hx)]}

    turn = inter.get("turn_ratio") or {}
    return {"position": pos, "length": length, "cross_length": cross_len,
            "nodes": nodes, "arms": arms,
            "turn_left": float(turn.get("left", 0.0)),
            "turn_right": float(turn.get("right", 0.0)),
            "vph": {"E": float(inter.get("opposite_vph", 0.0)),
                    "N": float(inter.get("cross_vph", 0.0)),
                    "S": float(inter.get("cross_vph", 0.0))},
            "ego_arm": "W"}


def _route_edges(layout, src: str, move: str, from_seg: int = 0):
    """src 팔에서 들어와 move(straight/left/right)로 나가는 엣지 나열."""
    arm = layout["arms"][src]
    dst = arm["to"][move]
    return [s["id"] for s in arm["segs"][from_seg:]] + [layout["arms"][dst]["out"]]


def _write_intersection_network(out_dir: str, layout: dict):
    """사거리용 nod/edg/con.xml 생성."""
    arms = layout["arms"]

    # ---- 노드: 교차로 중심 C만 priority 교차로, 나머지는 단순 연결점 ----
    node_lines = []
    for nid, (x, y) in layout["nodes"].items():
        ntype = ' type="priority"' if nid == "C" else ""
        node_lines.append(f'    <node id="{nid}" x="{x:.2f}" y="{y:.2f}"{ntype}/>')

    # ---- 엣지: 팔마다 접근로 구간들(차로 수 증가) + 진출로 1개 ----
    #   priority: 주도로(2) > 교차도로(1) → 신호 없는 교차로에서 주도로 우선.
    #   (ego는 주도로 직진이므로 멈출 일이 없고, 교차도로 차량이 양보한다)
    edge_lines = []
    for a, arm in arms.items():
        common = f'speed="{arm["speed"]}" priority="{arm["priority"]}"'
        for s in arm["segs"]:
            edge_lines.append(f'    <edge id="{s["id"]}" from="{s["from"]}" to="{s["to"]}" '
                              f'numLanes="{s["lanes"]}" {common}/>')
        edge_lines.append(f'    <edge id="{arm["out"]}" from="C" to="{a}0" '
                          f'numLanes="{arm["lanes"]}" {common}/>')

    # ---- 차선 연결: 확장 구간은 직접 지정해 "포켓이 옆 차로에서 갈라지는" 형태로 ----
    con = []

    def c(fe, fl, te, tl):
        con.append(f'    <connection from="{fe}" to="{te}" fromLane="{fl}" toLane="{tl}"/>')

    for a, arm in arms.items():
        n = arm["lanes"]
        segs = arm["segs"]
        # (a) 접근로 구간 → 다음 구간 (차로 수가 늘어나는 지점)
        for s0, s1 in zip(segs[:-1], segs[1:]):
            r0, l0, r1, l1 = s0["right"], s0["left"], s1["right"], s1["left"]
            for i in range(n):                                   # 직진 차로는 그대로
                c(s0["id"], r0 + i, s1["id"], r1 + i)
            for j in range(r0):                                  # 기존 우포켓 유지
                c(s0["id"], j, s1["id"], j)
            for j in range(l0):                                  # 기존 좌포켓 유지
                c(s0["id"], r0 + n + j, s1["id"], r1 + n + j)
            for j in range(r0, r1):                              # 새 우포켓 ← 맨 오른쪽 직진
                c(s0["id"], r0, s1["id"], j)
            for j in range(l0, l1):                              # 새 좌포켓 ← 맨 왼쪽 직진
                c(s0["id"], r0 + n - 1, s1["id"], r1 + n + j)
        # (b) 마지막 구간 → 교차로 C → 진출로
        last = segs[-1]
        r, l = last["right"], last["left"]
        straight = arms[arm["to"]["straight"]]["out"]
        right_out, right_n = arms[arm["to"]["right"]]["out"], arms[arm["to"]["right"]]["lanes"]
        left_out, left_n = arms[arm["to"]["left"]]["out"], arms[arm["to"]["left"]]["lanes"]
        for i in range(n):
            c(last["id"], r + i, straight, i)
        if r > 0:
            for j in range(r):
                c(last["id"], j, right_out, min(j, right_n - 1))
        else:                                    # 포켓 없으면 맨 오른쪽 차로가 직진+우회전 겸용
            c(last["id"], 0, right_out, 0)
        if l > 0:
            for j in range(l):
                c(last["id"], r + n + j, left_out, max(0, left_n - l + j))
        else:                                    # 포켓 없으면 맨 왼쪽 차로가 직진+좌회전 겸용
            c(last["id"], r + n - 1, left_out, left_n - 1)

    for name, tag, lines in (("nod", "nodes", node_lines),
                             ("edg", "edges", edge_lines),
                             ("con", "connections", con)):
        with open(os.path.join(out_dir, f"highway.{name}.xml"), "w") as f:
            f.write(f"<{tag}>\n" + "\n".join(lines) + f"\n</{tag}>\n")


def _intersection_demand(layout: dict, traffic: dict, flow_type: str) -> str:
    """사거리용 route/flow 블록 생성.

    - r0      : ego 경로 (서 → 동 직진). 기존 "traffic" flow도 이 경로의 직진 차량.
    - <팔>_<방향> : 각 진입로에서 직진/좌회전/우회전하는 배경 교통 flow.
                 회전 차량은 departLane="best" → 회전 가능한 차로로 진입.
    - pf_<엣지>  : 프리필 전용 경로 (그 엣지에서 출발해 직진으로 빠져나감).
    """
    arms = layout["arms"]
    tl, tr = layout["turn_left"], layout["turn_right"]
    vph = dict(layout["vph"], **{layout["ego_arm"]: float(traffic["vehs_per_hour"])})
    lines = [f'    <route id="r0" edges="{" ".join(_route_edges(layout, "W", "straight"))}"/>']
    for a in arms:
        for move in ("straight", "left", "right"):
            if not (a == "W" and move == "straight"):
                lines.append(f'    <route id="{a}_{move}" '
                             f'edges="{" ".join(_route_edges(layout, a, move))}"/>')
    for e in prefill_plan_routes(layout):
        lines.append(f'    <route id="{e["route"]}" edges="{" ".join(e["edges"])}"/>')

    for a in arms:
        for move, share in (("straight", 1.0 - tl - tr), ("left", tl), ("right", tr)):
            rate = vph.get(a, 0.0) * share
            if rate <= 0:
                continue
            fid = "traffic" if (a == "W" and move == "straight") else f"flow_{a}_{move}"
            route = "r0" if (a == "W" and move == "straight") else f"{a}_{move}"
            lane = traffic["depart_lane"] if move == "straight" else "best"
            lines.append(f'    <flow id="{fid}" type="{flow_type}" route="{route}" begin="0" end="100000"\n'
                         f'          period="{3600.0 / rate:.2f}"\n'
                         f'          departLane="{lane}" departSpeed="{traffic["depart_speed"]}"/>')
    return "\n".join(lines)


def prefill_plan_routes(layout: dict):
    """프리필 대상 엣지마다 '그 엣지에서 출발해 직진'하는 경로 정보.

    lanes: 직진 경로로 쓸 수 있는 차선 인덱스 (회전 포켓 차선은 제외).
    """
    arms = layout["arms"]
    out = []
    for a, arm in arms.items():
        for k, s in enumerate(arm["segs"]):
            out.append({"route": f"pf_{s['id']}", "src": a, "seg": k,
                        "edges": _route_edges(layout, a, "straight", k),
                        "length": s["length"],
                        "lanes": list(range(s["right"], s["right"] + arm["lanes"]))})
        # 진출로: 반대편 팔에서 직진해 들어온 교통이 채우는 구간
        feeder = next(b for b, o in arms.items() if o["to"]["straight"] == a)
        out.append({"route": f"pf_{arm['out']}", "src": feeder, "seg": None,
                    "edges": [arm["out"]], "length": arm["length"],
                    "lanes": list(range(arm["lanes"]))})
    return out


def prefill_plan(road: dict, traffic: dict):
    """sumo_env의 프리필이 사용할 배치 계획.

    반환: [{"route", "length", "lanes", "vph", "ego_start"}...]
      route    : traci.vehicle.add에 쓸 route id (출발 엣지 = 경로 첫 엣지)
      length   : 출발 엣지 길이 (m) — 이 범위 안에 departPos를 배치
      lanes    : 배치할 차선 인덱스 목록
      vph      : 해당 구간 교통량 (배치 간격 자동 계산용)
      ego_start: ego 출발 구간이면 True → 시작부 start_offset 만큼 비움
    """
    if osm_config(road) is not None:
        return []     # OSM 도로망은 프리필 대신 warmup으로 정체를 만든다
    layout = intersection_layout(road)
    vph_main = float(traffic.get("vehs_per_hour", 1800))
    if layout is None:
        return [{"route": "r0", "length": float(road["length"]),
                 "lanes": list(range(int(road["num_lanes"]))),
                 "vph": vph_main, "ego_start": True}]
    vph = dict(layout["vph"], **{layout["ego_arm"]: vph_main})
    plan = []
    for e in prefill_plan_routes(layout):
        plan.append({"route": e["route"], "length": e["length"], "lanes": e["lanes"],
                     "vph": vph.get(e["src"], 0.0),
                     "ego_start": e["src"] == layout["ego_arm"] and e["seg"] == 0})
    return plan


def _blob(cx: float, cy: float, r: float, rng, n: int = 14,
          roughness: float = 0.35) -> str:
    """유기적인(울퉁불퉁한) 수관 실루엣 다각형을 만든다.

    실제 나무를 위에서 내려다보면 완전한 원이 아니라 잎 뭉치들이
    만드는 불규칙한 테두리다. 꼭짓점마다 반지름을 무작위로 흔들되,
    이웃 꼭짓점과 평균을 내서(smoothing) 뾰족하지 않고 부드럽게
    울퉁불퉁한 실루엣을 얻는다.
    """
    import math
    raw = [1.0 + rng.uniform(-roughness, roughness) for _ in range(n)]
    sm = [(raw[i - 1] + raw[i] + raw[(i + 1) % n]) / 3.0 for i in range(n)]
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        rr = r * sm[i]
        pts.append(f"{cx + rr * math.cos(a):.2f},{cy + rr * math.sin(a):.2f}")
    return " ".join(pts)


def _write_scenery(out_dir: str, road: dict, layout=None):
    """도로변 나무를 highway.scenery.xml (additional 파일)로 생성.

    항공뷰(top-down) 조경처럼 보이도록 나무 한 그루를 여러 겹으로 그린다:
      layer 1  그림자  : 수관을 남동쪽으로 살짝 밀어낸 반투명 검정 blob
                         (해가 북서쪽에 있는 항공사진 느낌)
      layer 2  수관 몸통: 중간톤 초록의 불규칙 실루엣
      layer 3  잎 뭉치  : 수관 안에 어두운/짙은 클럼프 1~2개
                         → 캐노피의 입체적인 질감
      layer 4  하이라이트: 북서쪽으로 치우친 밝은 초록 작은 blob
    배치도 실제 조경처럼: 단독목 / 2~3그루 군락 / 낮은 관목이 섞인다.
    SCENERY["trees"]가 False면 빈 additional 파일 생성.
    """
    import random
    path = os.path.join(out_dir, "highway.scenery.xml")
    cfg = SCENERY if isinstance(SCENERY, dict) else {}
    if not cfg.get("trees", False):
        with open(path, "w") as f:
            f.write("<additional/>\n")
        return

    rng = random.Random(int(cfg.get("seed", 7)))
    spacing = float(cfg.get("tree_spacing", 45.0))
    jitter = float(cfg.get("tree_jitter", 0.5))
    side = float(cfg.get("side_offset", 6.0))
    length = float(road["length"])
    road_bottom = -3.2 * int(road["num_lanes"])

    # (몸통, 어두운 클럼프, 밝은 하이라이트) 초록 팔레트 — 수종별 톤
    palettes = [
        ("46,110,44", "32,84,32", "96,158,84"),
        ("38,98,40",  "26,72,28", "84,146,74"),
        ("56,122,50", "40,92,38", "110,168,92"),
        ("30,88,36",  "20,64,26", "76,138,70"),
    ]
    bush_greens = ["70,124,58", "84,136,64", "62,112,52"]

    lines = ["<additional>"]
    tid = [0]

    def draw_tree(x, y, r):
        body, dark, light = rng.choice(palettes)
        i = tid[0]; tid[0] += 1
        # 그림자 (남동쪽 +x, -y 로 오프셋, 반투명)
        lines.append(f'    <poly id="sh_{i}" color="20,26,18,70" fill="1" '
                     f'layer="1" shape="{_blob(x + r*0.35, y - r*0.35, r*1.02, rng)}"/>')
        # 수관 몸통
        lines.append(f'    <poly id="tr_{i}" color="{body}" fill="1" '
                     f'layer="2" shape="{_blob(x, y, r, rng)}"/>')
        # 잎 뭉치 1~2개 (어두운 톤, 수관 내부 무작위 위치)
        for k in range(rng.randint(1, 2)):
            ox = rng.uniform(-0.35, 0.35) * r
            oy = rng.uniform(-0.35, 0.35) * r
            lines.append(f'    <poly id="cl_{i}_{k}" color="{dark}" fill="1" '
                         f'layer="3" shape="{_blob(x + ox, y + oy, r * rng.uniform(0.4, 0.6), rng, n=10)}"/>')
        # 하이라이트 (북서쪽 -x, +y — 그림자 반대 방향)
        lines.append(f'    <poly id="hl_{i}" color="{light}" fill="1" '
                     f'layer="4" shape="{_blob(x - r*0.28, y + r*0.28, r*0.34, rng, n=8, roughness=0.25)}"/>')

    def draw_bush(x, y):
        i = tid[0]; tid[0] += 1
        g = rng.choice(bush_greens)
        r = rng.uniform(1.0, 1.8)
        lines.append(f'    <poly id="bsh_{i}" color="20,26,18,55" fill="1" '
                     f'layer="1" shape="{_blob(x + 0.3, y - 0.3, r, rng, n=8)}"/>')
        lines.append(f'    <poly id="bs_{i}" color="{g}" fill="1" '
                     f'layer="2" shape="{_blob(x, y, r, rng, n=8)}"/>')

    # 나무 줄(strip): (시작, 끝, 도로 기준선, 바깥 방향 부호, 세로 도로 여부)
    #   세로(vertical) 줄은 좌표를 (x, y) ↔ (y, x)로 바꿔서 그린다.
    if layout is None:
        strips = [(0.0, length, side, +1, False),
                  (0.0, length, road_bottom - side, -1, False)]
    else:
        # 사거리: 주도로는 양방향(서쪽행이 y>0), 교차로 부근은 비우고,
        #         교차도로(세로) 양옆에도 나무를 심는다.
        arms = layout["arms"]
        pos, cross_len = layout["position"], layout["cross_length"]
        top = LANE_WIDTH * arms["E"]["max_lanes"]           # 서쪽행 접근로 폭
        bottom = -LANE_WIDTH * arms["W"]["max_lanes"]       # 동쪽행 접근로 폭
        left = pos - LANE_WIDTH * arms["N"]["max_lanes"]     # 남쪽행 (x < pos)
        right = pos + LANE_WIDTH * arms["S"]["max_lanes"]    # 북쪽행 (x > pos)
        gap = side + 8.0
        strips = [
            (0.0, left - gap, top + side, +1, False),
            (right + gap, length, top + side, +1, False),
            (0.0, left - gap, bottom - side, -1, False),
            (right + gap, length, bottom - side, -1, False),
            (top + gap, cross_len, left - side, -1, True),
            (top + gap, cross_len, right + side, +1, True),
            (-cross_len, bottom - gap, left - side, -1, True),
            (-cross_len, bottom - gap, right + side, +1, True),
        ]

    for start, end, base_y, side_sign, vertical in strips:
        def plant(fn, a, b, *args):
            fn(b, a, *args) if vertical else fn(a, b, *args)

        x = start + rng.uniform(0, spacing)
        while x < end:
            y = base_y + side_sign * rng.uniform(0, 4.0)
            roll = rng.random()
            if roll < 0.12:
                plant(draw_bush, x, y)                        # 15%: 낮은 관목
            elif roll < 0.67:
                plant(draw_tree, x, y, rng.uniform(2.6, 4.6))  # 55%: 단독목 (실제 수관 스케일)
            else:
                # 25%: 2~3그루 군락 (겹치듯 붙여 심어 숲 덩어리 느낌)
                n = rng.randint(2, 4)
                for _ in range(n):
                    gx = x + rng.uniform(-5.5, 5.5)
                    gy = y + side_sign * rng.uniform(-2.0, 4.5)
                    plant(draw_tree, gx, gy, rng.uniform(2.2, 3.8))
                x += spacing * 0.4                     # 군락 뒤엔 간격 조금 더
            x += spacing * rng.uniform(1.0 - jitter, 1.0 + jitter)
    lines.append("</additional>")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
