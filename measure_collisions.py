"""학습된 모델을 화면 없이 주행시켜 충돌마다 상황을 CSV로 남기고 유형별로 집계한다.

사용법:
    python measure_collisions.py results/run_xxx/best_model.pt --episodes 50
    python measure_collisions.py results/run_xxx/best_model.pt --out collisions.csv

충돌 유형은 :func:`~utils.collision.classify`가 정한다.
"""
import argparse
import collections
import csv
import os

from env import road_builder
from env.mdp_config import ACTION, OBSERVATION, SIMULATION
from env.road_config import EGO, ROAD, TRAFFIC
from env.sumo_env import SumoHighwayEnv
from test import load_agent
from train import REWARD
from utils.collision import classify

BASE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    """명령줄 인자대로 평가 주행을 하고 충돌 기록 CSV와 유형별 집계를 출력한다."""
    parser = argparse.ArgumentParser(description="충돌 상황 측정")
    parser.add_argument("model", help="PPO model.pt 또는 best_model.pt 경로")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--out", default="collisions.csv", help="충돌 기록 CSV 경로")
    args = parser.parse_args()

    sumocfg = road_builder.build(ROAD, EGO, TRAFFIC, os.path.join(BASE, "env", "sumo"))
    env = SumoHighwayEnv(cfg_path=sumocfg, road=ROAD, ego=EGO,
                         mdp_sim=SIMULATION, mdp_obs=OBSERVATION,
                         action=ACTION, reward=REWARD, traffic=TRAFFIC, gui=False)
    records = []
    endings = collections.Counter()
    try:
        agent = load_agent(args.model, env.observation_space.shape[0],
                           env.action_space.shape[0])
        for ep in range(args.episodes):
            obs, _ = env.reset()
            done, ended_by = False, "timeout"
            while not done:
                obs, _, terminated, truncated, info = env.step(
                    agent.predict(obs, deterministic=True))
                done = terminated or truncated
                if info["collision"] is not None:
                    record = {"episode": ep, **info["collision"]}
                    record["category"] = classify(record)
                    records.append(record)
                    ended_by = "collision"
                elif info["red_light"]:
                    ended_by = "red_light"
                elif info["exit_missed"]:
                    ended_by = "exit_miss"
                elif info["arrived"]:
                    ended_by = "arrival"
            endings[ended_by] += 1
            print(f"episode {ep + 1}/{args.episodes}: {ended_by}", flush=True)
    finally:
        env.close()

    if records:
        fields = list(dict.fromkeys(k for r in records for k in r))
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    print(f"\n종료 원인 ({args.episodes} 에피소드): {dict(endings)}")
    print(f"충돌 {len(records)}건 유형별:")
    for name, n in collections.Counter(r["category"] for r in records).most_common():
        print(f"  {name:28s} {n:3d}  ({n / len(records):.0%})")
    print(f"기록: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
