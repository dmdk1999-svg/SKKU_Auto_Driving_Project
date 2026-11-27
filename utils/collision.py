"""충돌 기록의 유형 분류.

충돌 기록은 :meth:`~env.sumo_env.SumoHighwayEnv.step`이 ``info["collision"]``으로 넘기는
:class:`dict`다.
"""

# 차선변경을 끝낸 뒤 이 스텝 수 안에 뒤에서 받히면 끼어들기 직후 추돌로 본다.
CUTIN_WINDOW_STEPS = 10


def classify(record: dict) -> str:
    """충돌 기록 ``record``의 유형 이름을 반환한다.

    분기점에서 다른 가지 쪽에 걸쳐 충돌로 처리한 기록은 ``diverge_straddle``이다.
    교차로 안 충돌을 먼저 가르고, 그다음 차선변경 기동 중 충돌을 가른다. 나머지는
    ego가 가해 차량이면 앞차 추돌, 피해 차량이면 뒤차에 받힘으로 나눈다. 뒤차에 받힌
    충돌은 차선변경 완료 후 :data:`CUTIN_WINDOW_STEPS` 스텝 안이면 끼어들기 직후로 본다.
    SUMO가 충돌 상대를 알려 주지 않은 기록은 ``unknown``이다.
    """
    if record.get("diverge_straddle"):
        return "diverge_straddle"
    if record.get("role") is None:
        return "unknown"
    if record["in_junction"]:
        return "junction"
    if record["lc_phase"] != "none":
        return f"lane_change_{record['lc_phase']}"
    if record["role"] == "collider":
        return "ego_rear_ends_leader"
    since = record["steps_since_lc"]
    if since is not None and since <= CUTIN_WINDOW_STEPS:
        return "rear_ended_after_cutin"
    return "rear_ended"
