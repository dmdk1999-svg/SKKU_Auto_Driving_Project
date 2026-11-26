"""Train and compare three PPO initializations. Run from the team repo root."""

import argparse
import csv
import json
import math
import shutil
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from time import monotonic

import numpy as np
import torch
import traci
from sumolib import checkBinary
from torch.nn import functional as F

REPO = Path.cwd()
if not (REPO / "songdo_C" / "songdo_intersection" / "sessions").is_dir():
    raise SystemExit("Run this script from the SKKU_Auto_Driving_Project directory.")
sys.path.insert(0, str(REPO))

from algorithms.ppo import PPO
from env import road_builder
from env.mdp_config import ACTION, OBSERVATION, SIMULATION
from env.road_config import EGO, ROAD, TRAFFIC
from env.sumo_env import SumoHighwayEnv
from train import HPARAMS, REWARD, TOTAL_TIMESTEPS
from utils.buffer import RolloutBuffer
from utils.logger import RunLogger
from utils.networks import GaussianActorCritic

TRAIN_SESSIONS = ("AM1", "AM2", "AM3", "AM4", "AM5", "PM1", "PM2")
VALID_SESSIONS = ("PM3",)
VARIANTS = ("full_bc", "loose_bc", "scratch")


class SeededHannam(SumoHighwayEnv):
    """Use a fixed simulator-seed sequence across variants."""

    def __init__(self, cfg, seed, route, sim):
        super().__init__(
            str(cfg), ROAD, EGO, sim, OBSERVATION, ACTION, REWARD, traffic=TRAFFIC
        )
        self.episode_seeds = np.random.default_rng(seed)
        if route:
            self.ego_routes = [route]

    def reset(self, seed=None, options=None):
        if seed is None:
            seed = int(self.episode_seeds.integers(0, 2**32))
        old_state = np.random.get_state()
        try:
            # The team env uses legacy np.random for both SUMO seed and ego route.
            np.random.seed(seed)
            obs, info = super().reset(seed=seed, options=options)
        finally:
            np.random.set_state(old_state)
        info["route"] = traci.vehicle.getRouteID(self.ego_id)
        return obs, info


def eta(seconds):
    return (datetime.now().astimezone() + timedelta(seconds=seconds)).strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )


def atomic_torch_save(state, path):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


class CheckpointLogger(RunLogger):
    def __init__(self, directory, agent, env, args, remaining_models):
        super().__init__(str(directory))
        self.directory, self.agent, self.env = directory, agent, env
        self.interval = args.checkpoint_every
        self.goal = math.ceil(args.steps / agent.n_steps) * agent.n_steps
        self.remaining_models = remaining_models
        self.initial_step, self.start = agent.num_timesteps, monotonic()
        log = directory / "training_log.csv"
        if self.initial_step and log.is_file():
            with log.open(encoding="utf-8", newline="") as file:
                self.train_rows = [
                    row for row in csv.DictReader(file) if int(row["steps"]) <= self.initial_step
                ]

    def log_train(self, step, metrics, verbose=True):
        update = step // self.agent.n_steps
        super().log_train(step, {**metrics, "update": update}, verbose=verbose)
        elapsed = monotonic() - self.start
        rate = (step - self.initial_step) / max(elapsed, 1e-9)
        remaining = self.goal - step
        all_remaining = remaining + self.remaining_models * self.goal
        print(
            f"{self.directory.name} {step:,}/{self.goal:,} | {rate:.2f} steps/s | "
            f"model ETA {eta(remaining / rate)} | "
            f"all PPO ETA {eta(all_remaining / rate)} (evaluation excluded)",
            flush=True,
        )
        if update % self.interval == 0 or step >= self.goal:
            atomic_torch_save({
                "policy": self.agent.policy.state_dict(),
                "optimizer": self.agent.optimizer.state_dict(),
                "steps": step,
                "traffic_rng": self.env.episode_seeds.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": (
                    torch.cuda.get_rng_state(self.agent.device)
                    if self.agent.device.type == "cuda" else None
                ),
            }, self.directory / "resume.pt")
            self.save_csv()
            if self.writer is not None:
                self.writer.flush()


def collect_songdo(session, args, limit=None):
    """Label reactive SUMO teacher motion using the team observation and units."""
    cfg = REPO / "songdo_C" / "songdo_intersection" / "sessions" / session / "scenario.sumocfg"
    observer = SumoHighwayEnv(str(cfg), ROAD, EGO, SIMULATION, OBSERVATION, ACTION, REWARD)
    dt = observer.step_length
    # A 0.1 s physics tick preserves the Songdo teacher's 0.6 s reaction period
    # and produces exact 0.5 s supervised transitions for the team policy.
    ticks_per_label = 5
    command = [
        checkBinary("sumo"), "-c", str(cfg), "--step-length", str(dt / ticks_per_label),
        "--seed", str(args.seed),
        "--no-step-log", "true", "--no-warnings", "true", "--end", "2400",
    ]
    rng = np.random.default_rng(args.seed)
    observations, actions = [], []
    step = 0
    traci.start(command, numRetries=3)
    try:
        while traci.simulation.getMinExpectedNumber() and traci.simulation.getTime() < 2400:
            sampled = {}
            if step % args.sample_every == 0:
                ids = sorted(traci.vehicle.getIDList())
                if len(ids) > args.cars_per_sample:
                    ids = rng.choice(ids, args.cars_per_sample, replace=False)
                for vid in ids:
                    observer.ego_id = str(vid)
                    sampled[vid] = (
                        observer._get_obs(),
                        traci.vehicle.getRoadID(vid),
                        traci.vehicle.getSpeed(vid),
                        traci.vehicle.getLateralSpeed(vid),
                    )
            collided = set()
            for _ in range(ticks_per_label):
                traci.simulationStep()
                if sampled:
                    collided.update(traci.simulation.getCollidingVehiclesIDList())
            step += 1
            if not sampled:
                continue
            # Preserve the sampling order for determinism across Python processes.
            alive = set(traci.vehicle.getIDList()) - collided
            for vid, (obs, edge, speed, lateral_speed) in sampled.items():
                if vid not in alive or traci.vehicle.getRoadID(vid) != edge:
                    continue
                acceleration = (traci.vehicle.getSpeed(vid) - speed) / dt
                lateral_accel = (traci.vehicle.getLateralSpeed(vid) - lateral_speed) / dt
                if edge.startswith(":"):
                    lateral_accel = 0.0
                scale = observer.max_accel if acceleration >= 0 else observer.max_decel
                observations.append(obs)
                actions.append(np.clip(
                    [acceleration / scale, lateral_accel / observer.max_lat_accel], -1.0, 1.0
                ))
                if limit and len(actions) >= limit:
                    break
            if limit and len(actions) >= limit:
                break
    finally:
        traci.close()
    if not observations:
        raise RuntimeError(f"{session}: no usable Songdo demonstrations")
    print(f"Songdo {session}: {len(observations):,} pairs", flush=True)
    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def load_demonstrations(out, args):
    destination = out / "songdo_43d.npz"
    train_sessions = TRAIN_SESSIONS[:1] if args.smoke else TRAIN_SESSIONS
    limit = 96 if args.smoke else None
    train = [collect_songdo(session, args, limit) for session in train_sessions]
    valid = [collect_songdo(session, args, limit) for session in VALID_SESSIONS]
    arrays = [np.concatenate(values) for values in (*zip(*train), *zip(*valid))]
    obs, actions, val_obs, val_actions = arrays
    if obs.shape[1:] != (43,) or actions.shape[1:] != (2,):
        raise RuntimeError(f"Expected observations/actions (N,43)/(N,2): {obs.shape}, {actions.shape}")
    if any(not np.isfinite(values).all() for values in arrays):
        raise RuntimeError("Songdo demonstrations contain non-finite observations or actions.")
    np.savez_compressed(
        destination, observation=obs, action_raw=actions, val_observation=val_obs,
        val_action_raw=val_actions, train_sessions=train_sessions, val_sessions=VALID_SESSIONS,
    )
    return arrays


def fit_bc(out, args, hp):
    xs, ys, val_xs, val_ys = map(torch.from_numpy, load_demonstrations(out, args))
    torch.manual_seed(args.seed)
    policy = GaussianActorCritic(
        43, 2, hidden_sizes=hp["hidden_sizes"], activation=hp["activation"],
        init_log_std=hp["init_log_std"],
    )
    actor_params = list(policy.trunk.parameters()) + list(policy.mu_head.parameters())
    optimizer = torch.optim.Adam(actor_params, lr=3e-4)
    generator = torch.Generator().manual_seed(args.seed)
    losses = []
    best_loss = float("inf")
    started = monotonic()
    for epoch in range(1, args.bc_epochs + 1):
        policy.train()
        for batch in torch.randperm(len(xs), generator=generator).split(256):
            mean, _, _ = policy(xs[batch])
            loss = F.mse_loss(mean, ys[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        policy.eval()
        with torch.no_grad():
            prediction, _, _ = policy(val_xs)
            val_loss = float(F.mse_loss(prediction, val_ys))
        losses.append({"epoch": epoch, "val_mse": val_loss})
        if epoch == args.loose_epochs:
            atomic_torch_save(policy.state_dict(), out / "loose_bc_init.pt")
        # Keep the full candidate strictly later than the loose checkpoint.
        if epoch > args.loose_epochs and val_loss < best_loss:
            atomic_torch_save(policy.state_dict(), out / "full_bc_init.pt")
            best_loss = val_loss
        if epoch <= 2 or epoch % 5 == 0 or epoch == args.bc_epochs:
            remaining = (monotonic() - started) / epoch * (args.bc_epochs - epoch)
            print(
                f"BC epoch {epoch}/{args.bc_epochs}: validation MSE {val_loss:.6f} | "
                f"BC ETA {eta(remaining)}", flush=True,
            )
    save_json(out / "bc_validation.json", losses)


def save_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def evaluate(agent, cfg, args, seed_list, phase, variant):
    results = []
    sim = dict(SIMULATION, max_steps=10) if args.smoke else SIMULATION
    env = SeededHannam(cfg, args.seed, args.route, sim)
    if args.smoke:
        env.warmup_steps = 20
    try:
        for seed in seed_list:
            obs, initial = env.reset(seed=seed)
            terminated = truncated = False
            total = 0.0
            changes = 0
            reached_edge = False
            while not (terminated or truncated):
                obs, reward, terminated, truncated, info = env.step(agent.predict(obs))
                total += reward
                changes += int(info["lane_change_applied"])
                if args.target_edge and not terminated:
                    reached_edge |= traci.vehicle.getRoadID(env.ego_id) == args.target_edge
            outcome = next(
                (label for key, label in (
                    ("collided", "collision"), ("red_light", "red_light"), ("arrived", "arrival")
                ) if info[key]),
                "timeout" if truncated and not terminated else "removed",
            )
            results.append({
                "variant": variant, "phase": phase, "traffic_seed": seed,
                "route": initial["route"], "outcome": outcome, "return": total,
                "steps": env.step_count, "lane_changes": changes,
                "target_edge_reached": reached_edge if args.target_edge else None,
            })
    finally:
        env.close()
    return results


def train_variant(variant, cfg, out, args, hp, seeds, remaining_models):
    agent = PPO(obs_dim=43, act_dim=2, buffer=RolloutBuffer(), **hp)
    if variant != "scratch":
        state = torch.load(out / f"{variant}_init.pt", weights_only=True, map_location="cpu")
        agent.policy.trunk.load_state_dict({
            key.removeprefix("trunk."): value for key, value in state.items() if key.startswith("trunk.")
        })
        agent.policy.mu_head.load_state_dict({
            key.removeprefix("mu_head."): value for key, value in state.items() if key.startswith("mu_head.")
        })
    directory = out / variant
    directory.mkdir(exist_ok=True)
    evaluation = directory / "evaluation.json"
    before = directory / "before_ppo.json"
    model = directory / "model.pt"
    checkpoint = directory / "resume.pt"
    goal = math.ceil(args.steps / agent.n_steps) * agent.n_steps
    if model.is_file() and evaluation.is_file():
        print(f"Reusing completed {variant}", flush=True)
        return json.loads(evaluation.read_text(encoding="utf-8")), goal
    if before.is_file():
        rows = json.loads(before.read_text(encoding="utf-8"))
    else:
        rows = evaluate(agent, cfg, args, seeds, "before_ppo", variant)
        save_json(before, rows)
    env = SeededHannam(cfg, args.seed, args.route, SIMULATION)
    if args.smoke:
        env.warmup_steps = 20
    if checkpoint.is_file():
        state = torch.load(checkpoint, weights_only=True, map_location="cpu")
        agent.policy.load_state_dict(state["policy"])
        agent.optimizer.load_state_dict(state["optimizer"])
        agent.num_timesteps = state["steps"]
        env.episode_seeds.bit_generator.state = state["traffic_rng"]
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None and agent.device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], agent.device)
        print(f"Resuming {variant} from {agent.num_timesteps:,} steps with a fresh episode", flush=True)
    else:
        print(f"Starting {variant} with {goal:,} PPO steps", flush=True)
    logger = CheckpointLogger(directory, agent, env, args, remaining_models)
    try:
        if agent.num_timesteps < goal:
            agent.learn(env, total_timesteps=args.steps, logger=logger)
        atomic_torch_save(agent.policy.state_dict(), model)
    finally:
        env.close()
        logger.close()
    rows.extend(evaluate(agent, cfg, args, seeds, "after_ppo", variant))
    save_json(evaluation, rows)
    print(f"Finished {variant}: {agent.num_timesteps:,} PPO steps", flush=True)
    return rows, agent.num_timesteps


def parse_args():
    routes = ROAD["osm"]["ego_routes"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    parser.add_argument("--variant", choices=("all", *VARIANTS), default="all")
    parser.add_argument("--route", choices=routes, help="Pin an existing Hannam ego route")
    parser.add_argument("--target-edge", help="Optional SUMO edge ID on the pinned route")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--bc-epochs", type=int, default=50)
    parser.add_argument("--loose-epochs", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--sample-every", type=int, default=2)
    parser.add_argument("--cars-per-sample", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint-every", type=int, default=5, help="PPO updates per checkpoint")
    parser.add_argument("--resume", action="store_true", help="Resume the same --out with the same settings")
    parser.add_argument("--bc-only", action="store_true", help="Train both Songdo BC initializations only")
    parser.add_argument("--bc-from", type=Path, help="Reuse compatible BC files from a completed run")
    parser.add_argument("--smoke", action="store_true", help="Validate the whole pipeline quickly")
    args = parser.parse_args()
    if args.smoke:
        args.bc_epochs, args.loose_epochs, args.steps, args.eval_episodes = 2, 1, 8, 1
    if not 0 <= args.seed < 2**32 or min(
        args.steps, args.eval_episodes, args.sample_every, args.cars_per_sample, args.checkpoint_every
    ) < 1:
        parser.error("seed must be nonnegative and step/sample counts must be positive")
    if not 1 <= args.loose_epochs < args.bc_epochs:
        parser.error("require 1 <= --loose-epochs < --bc-epochs")
    if args.target_edge and not args.route:
        parser.error("--target-edge requires --route")
    if not ROAD["osm"]["enabled"]:
        parser.error("Enable the team OSM road configuration before running.")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(2)
    hp = dict(HPARAMS, seed=args.seed, device=args.device)
    if not args.bc_only and args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is unavailable. Activate a CUDA-enabled environment.")
    if hp["policy_type"] != "gaussian":
        raise ValueError("This experiment requires the team's 2-axis Gaussian PPO.")
    if args.smoke:
        hp.update(n_steps=8, n_epochs=1, minibatch_size=4)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    seeds = [int(seed) for seed in np.random.default_rng(20261125).integers(
        0, 2**32, size=args.eval_episodes
    )]
    recorded = out / "settings.json"
    settings = {
        **vars(args), "out": str(out), "bc_from": str(args.bc_from.resolve()) if args.bc_from else None,
        "ppo": hp, "evaluation_seeds": seeds, "reward": REWARD, "simulation": SIMULATION,
        "road": ROAD, "ego": EGO, "traffic": TRAFFIC, "observation": OBSERVATION, "action": ACTION,
    }
    settings.pop("resume")
    settings = json.loads(json.dumps(settings))
    if args.resume and recorded.is_file():
        if json.loads(recorded.read_text(encoding="utf-8")) != settings:
            raise ValueError("Resume requires the original settings and unchanged team configuration.")
    else:
        save_json(recorded, settings)
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    if args.bc_only or any(variant != "scratch" for variant in variants):
        prepare_bc(out, args, settings)
    if args.bc_only:
        return
    network = out / "sumo"
    cfg = road_builder.build(ROAD, EGO, TRAFFIC, str(network))
    if args.target_edge:
        from env.osm_road import resolve_routes
        routes = resolve_routes(str(network / "highway.net.xml"), ROAD["osm"])
        if args.target_edge not in routes[args.route]:
            raise ValueError(f"{args.target_edge} is not on Hannam route {args.route}")
    summaries = []
    for index, variant in enumerate(variants):
        rows, steps = train_variant(variant, cfg, out, args, hp, seeds, len(variants) - index - 1)
        for phase in ("before_ppo", "after_ppo"):
            group = [row for row in rows if row["phase"] == phase]
            counts = Counter(row["outcome"] for row in group)
            summary = {
                "variant": variant, "phase": phase, "ppo_steps": steps if phase == "after_ppo" else 0,
                "episodes": len(group), "arrival_rate": counts["arrival"] / len(group),
                "collision_rate": counts["collision"] / len(group),
                "red_light_rate": counts["red_light"] / len(group),
                "timeout_rate": counts["timeout"] / len(group),
                "removed_rate": counts["removed"] / len(group),
                "mean_return": sum(row["return"] for row in group) / len(group),
                "mean_lane_changes": sum(row["lane_changes"] for row in group) / len(group),
                "target_edge_rate": (
                    sum(row["target_edge_reached"] for row in group) / len(group)
                    if args.target_edge else None
                ),
            }
            summaries.append(summary)
        # Preserve completed comparisons if a later variant is interrupted.
        with (out / "comparison.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)


def prepare_bc(out, args, settings):
    files = ("full_bc_init.pt", "loose_bc_init.pt", "bc_validation.json")
    if all((out / file).is_file() for file in files):
        print("Reusing completed Songdo BC", flush=True)
        return
    if not args.bc_from:
        fit_bc(out, args, settings["ppo"])
        return
    source = args.bc_from.resolve()
    old = json.loads((source / "settings.json").read_text(encoding="utf-8"))
    keys = ("seed", "smoke", "bc_epochs", "loose_epochs", "sample_every", "cars_per_sample")
    if any(old[key] != settings[key] for key in keys):
        raise ValueError("BC source settings must match seed, sampling and BC epoch settings.")
    for key in ("hidden_sizes", "activation", "policy_type"):
        if old["ppo"][key] != settings["ppo"][key]:
            raise ValueError(f"BC source has a different PPO {key}.")
    for key in ("ego", "observation", "action"):
        if key in old and old[key] != settings[key]:
            raise ValueError(f"BC source has a different {key} configuration.")
    for file in files:
        temporary = (out / file).with_suffix(".tmp")
        shutil.copyfile(source / file, temporary)
        temporary.replace(out / file)
    print(f"Reusing Songdo BC from {source}", flush=True)


if __name__ == "__main__":
    main()
