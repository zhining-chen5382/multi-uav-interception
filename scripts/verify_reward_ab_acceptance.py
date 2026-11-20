"""Check production A/B state transitions against saved physical trajectories."""

import argparse
from copy import deepcopy
from decimal import Decimal, localcontext
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from algorithms.ppo import PPOAgent
from envs.interception_env import InterceptionEnv
from scripts.offline_best_so_far_reward_replay import best_so_far_shaping
from utils.experiment import set_seed
from utils.formal_ppo_config import load_formal_config, validate_readiness
from utils.formal_ppo_evaluation import write_json
from scripts.offline_near_reward_replay import read_csv, sha256


ROOT = Path(__file__).resolve().parents[1]


def verify(config_path, baseline_root, output_dir):
    config = load_formal_config(config_path)
    validate_readiness(config, expected_seed_count=3)
    baseline_root, output_dir = Path(baseline_root).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("Use a new verification directory")
    original_environment = yaml.safe_load((baseline_root / "seed_1001/environment_config.yaml").read_text(encoding="utf-8"))
    stripped = deepcopy(config["environment_config"])
    shaping_config = stripped["reward"].pop("best_so_far_shaping")
    if stripped != original_environment:
        raise ValueError("B changes physical, observation, normalization or original reward settings")
    initials = []
    for seed in config["experiment"]["seeds"]:
        path = baseline_root / ("seed_%d" % seed) / "checkpoints/initial.pt"
        original = torch.load(path, map_location="cpu", weights_only=False)
        for name in ("network", "ppo", "training"):
            if config[name] != original["config"][name]:
                raise ValueError("B changes the original protocol " + name)
        set_seed(seed, config["torch_num_threads"])
        fresh = PPOAgent(config)
        matches = {name: all(torch.equal(value, original["agent"][name][key])
                            for key, value in getattr(fresh, name).state_dict().items())
                   for name in ("actor", "critic")}
        if not all(matches.values()):
            raise ValueError("Fresh B initialization differs from same-seed A initialization")
        if fresh.actor_optimizer.state or fresh.critic_optimizer.state or fresh.update_count:
            raise ValueError("B must begin with fresh Adam and zero updates")
        initials.append(dict(seed=seed, independently_initialized=True,
                             actor_bitwise_matches_historical_a=matches["actor"],
                             critic_bitwise_matches_historical_a=matches["critic"],
                             empty_optimizer_states=True, historical_initial_sha256=sha256(path)))
    representatives_path = ROOT / "outputs/formal_ppo_v1/_离线奖励标定/01_普通距离差分/representatives.csv"
    rows = []
    for source in read_csv(representatives_path):
        path = Path(source["trajectory"])
        before = sha256(path)
        with np.load(path, allow_pickle=False) as archive:
            states, actions, recorded = archive["states"], archive["actions"], archive["rewards"]
            metadata = json.loads(archive["metadata_json"].item())
        env_a = InterceptionEnv(original_environment, seed=metadata["seed"])
        env_b = InterceptionEnv(config["environment_config"], seed=metadata["seed"])
        observation_a, _ = env_a.reset(seed=metadata["seed"], angle_degrees=metadata["angle_degrees"])
        observation_b, _ = env_b.reset(seed=metadata["seed"], angle_degrees=metadata["angle_degrees"])
        np.testing.assert_array_equal(observation_a, observation_b)
        np.testing.assert_array_equal(env_b.states, states[0])
        near, rewards = [], []
        for index, action in enumerate(actions):
            a = env_a.step(action)
            b = env_b.step(action)
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(env_a.states, env_b.states)
            np.testing.assert_array_equal(env_b.states, states[index + 1])
            np.testing.assert_allclose(a[1], recorded[index], rtol=0, atol=1e-10)
            assert a[2:4] == b[2:4]
            for name, value in a[4]["reward_components"].items():
                assert value == b[4]["reward_components"][name]
            near.append(b[4]["reward_components"]["near_shaping"])
            rewards.append(b[1])
        distances = np.linalg.norm(states[:, 0, :3] - states[:, 1, :3], axis=1)
        expected = best_so_far_shaping(distances, shaping_config["length_scale_m"], shaping_config["beta"])
        # NumPy scalar and SIMD vector exp can differ by a few float64 ulps.
        # An independent high-precision reference checks both evaluations.
        with localcontext() as context:
            context.prec = 60
            scale = Decimal.from_float(float(shaping_config["length_scale_m"]))
            beta = Decimal.from_float(float(shaping_config["beta"]))
            potential = [(-Decimal.from_float(float(distance)) / scale).exp()
                         for distance in np.minimum.accumulate(distances)]
            decimal_reference = np.asarray([float(beta * (right - left))
                                            for left, right in zip(potential, potential[1:])])
        np.testing.assert_allclose(near, expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(near, decimal_reference, rtol=0, atol=1e-12)
        np.testing.assert_allclose(expected, decimal_reference, rtol=0, atol=1e-12)
        np.testing.assert_allclose(rewards, recorded + expected, rtol=0, atol=1e-10)
        assert before == sha256(path)
        rows.append(dict(episode_id=source["episode_id"], category=source["category"],
            trajectory=str(path), source_sha256=before, steps=len(actions),
            physical_states_bitwise_match=True, observations_bitwise_match=True,
            original_reward_preserved=True, online_shaping_matches_offline_within_1e_minus_12=True,
            max_online_offline_reward_error=float(np.max(np.abs(np.asarray(near) - expected))),
            max_online_decimal_reward_error=float(np.max(np.abs(np.asarray(near) - decimal_reference))),
            original_return=float(recorded.sum()), near_shaping_return=float(expected.sum()),
            shaped_return=float(np.sum(rewards)), terminal_flags_match=True))
    output_dir.mkdir(parents=True, exist_ok=False)
    result = dict(passed=True, settings="Production B parameters, historical A300k, saved complete physical trajectories",
                  fresh_initializations=initials, real_trajectory_replays=rows,
                  environment_changes_only_best_so_far_shaping=True,
                  shaping_config=shaping_config, representatives_sha256=sha256(representatives_path))
    write_json(output_dir / "REAL_TRAJECTORY_AND_INITIALIZATION_CHECK.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ppo_reward_ab.yaml")
    parser.add_argument("--baseline-root", type=Path, default=ROOT / "outputs/formal_ppo_v1/v14_单机PPO_原始奖励/01_从零训练至300k")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    outcome = verify(args.config, args.baseline_root, args.output_dir)
    print(json.dumps(dict(passed=outcome["passed"], seeds=len(outcome["fresh_initializations"]),
                         real_trajectories=len(outcome["real_trajectory_replays"])), ensure_ascii=False))
