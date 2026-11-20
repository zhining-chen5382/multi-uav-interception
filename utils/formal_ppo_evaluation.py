"""Shared held-out scenarios and deterministic evaluation for the 1v1 PPO flow."""

from contextlib import contextmanager
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
import yaml

from envs.interception_env import InterceptionEnv


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def write_csv(path, rows):
    if not rows:
        raise ValueError('At least one row is required')
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scenario_hash(angles):
    return hashlib.sha256(np.asarray(angles, dtype='<f8').tobytes()).hexdigest()


def create_scenarios(config, environment, output_dir):
    """Freeze one validation grid and one independent random final-test set."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    lower, upper = environment['initialization']['attacker_angle_range_degrees']
    validation = np.linspace(lower, upper, config['training']['validation_episodes'])
    generator = np.random.default_rng(config['experiment']['test_seed'])
    test = generator.uniform(lower, upper, size=config['experiment']['test_episodes'])
    while len(set(map(float, test))) != len(test) or np.isin(test, validation).any():
        test = generator.uniform(lower, upper, size=config['experiment']['test_episodes'])
    for name, angles, base_seed in (
        ('validation', validation, config['experiment']['validation_seed']),
        ('test', test, config['experiment']['test_seed']),
    ):
        write_csv(directory / (name + '_scenarios.csv'), [
            dict(scenario_id=index, angle_degrees=float(angle), environment_seed=int(base_seed + index))
            for index, angle in enumerate(angles)
        ])
    manifest = dict(validation_angles=validation.tolist(), test_angles=test.tolist(),
                    validation_sha256=scenario_hash(validation), test_sha256=scenario_hash(test),
                    test_seed=config['experiment']['test_seed'], validation_seed=config['experiment']['validation_seed'],
                    training_seeds=config['experiment']['seeds'],
                    test_generation='Independent NumPy Generator uniform distribution; shared by all five frozen models',
                    validation_generation='Linearly spaced angles including both endpoints',
                    exact_validation_test_overlap=False)
    write_json(directory / 'scenario_manifest.json', manifest)
    return manifest


@contextmanager
def evaluation_context(agent):
    """Preserve global RNG streams and the training/evaluation modes."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    actor_mode, critic_mode = agent.actor.training, agent.critic.training
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            agent.actor.eval()
            agent.critic.eval()
            with torch.inference_mode():
                yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        agent.actor.train(actor_mode)
        agent.critic.train(critic_mode)


def wilson_interval(successes, count):
    z = 1.959963984540054
    fraction = successes / count
    divisor = 1 + z * z / count
    center = (fraction + z * z / (2 * count)) / divisor
    half = z * np.sqrt(fraction * (1 - fraction) / count + z * z / (4 * count * count)) / divisor
    return [float(max(0, center - half)), float(min(1, center + half))]


def summarize(rows):
    captured = [row for row in rows if row['captured']]
    result = dict(episodes=len(rows), captures=len(captured), capture_rate=len(captured) / len(rows),
                defense_failure_rate=float(np.mean([row['defense_failure'] for row in rows])),
                timeout_rate=float(np.mean([row['timeout'] for row in rows])),
                average_interception_time=float(np.mean([r['capture_time'] for r in captured])) if captured else None,
                average_min_distance=float(np.mean([r['min_distance'] for r in rows])),
                average_return=float(np.mean([r['episode_return'] for r in rows])),
                average_episode_length=float(np.mean([r['episode_length'] for r in rows])),
                capture_rate_wilson_95=wilson_interval(len(captured), len(rows)))
    if 'line_cost_return' in rows[0]:
        for field in ('original_return', 'near_shaping_return', 'line_cost_return',
                      'return_without_line_cost', 'line_segment_distance'):
            result['average_' + field] = float(np.mean([row.get(field, 0.0) for row in rows]))
    if 'near_shaping_uncorrected_return' in rows[0]:
        for field in ('original_return', 'near_shaping_return', 'near_shaping_uncorrected_return',
                      'near_shaping_removed_return', 'near_counterfactual_distance'):
            result['average_' + field] = float(np.mean([row[field] for row in rows]))
    return result


def evaluate_angles(agent, env_config, angles, seed, output_dir):
    """Evaluate complete episodes with batched inference and independent envs.

    All scenes use the exact same environment normalization contract as training.
    Evaluation does not touch a collector or a training rollout buffer.
    """
    angles = np.asarray(angles, dtype=float)
    if angles.ndim != 1 or not len(angles) or not np.isfinite(angles).all():
        raise ValueError('Evaluation angles must be a finite nonempty vector')
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    trajectories_dir = directory / 'trajectories'
    trajectories_dir.mkdir()
    with (directory / 'environment_config.yaml').open('x', encoding='utf-8') as stream:
        yaml.safe_dump(env_config, stream, sort_keys=False, allow_unicode=True)
    envs = [InterceptionEnv(deepcopy(env_config), seed=int(seed + index)) for index in range(len(angles))]
    observations, histories = [], []
    for index, (environment, angle) in enumerate(zip(envs, angles)):
        observation, info = environment.reset(seed=int(seed + index), angle_degrees=float(angle))
        observations.append(observation)
        histories.append(dict(states=[environment.states.copy()], observations=[observation.copy()],
                              raw_observations=[np.asarray(info['raw_observation']).copy()],
                              actions=[], rewards=[], applied_accelerations=[], realized_accelerations=[],
                              normalization_violations=[]))
        if 'near_shaping' in environment.reward_fields:
            histories[-1].update(best_distances=[float(info['best_distance_so_far'])], near_shaping=[])
        if 'line_cost' in environment.reward_fields:
            histories[-1].update(line_distances=[float(info['line_segment_distance'])], line_cost=[])
        if 'near_shaping_uncorrected' in info:
            histories[-1].update(near_shaping_uncorrected=[], near_shaping_removed=[],
                                 near_counterfactual_distances=[float(info['near_counterfactual_distance'])])
    active = list(range(len(envs)))
    rows = [None] * len(envs)
    with evaluation_context(agent):
        while active:
            sampled = agent.act(np.asarray([observations[index] for index in active]), deterministic=True)
            next_active = []
            for action_index, index in enumerate(active):
                environment, history = envs[index], histories[index]
                action = sampled['actions'][action_index]
                observation, reward, terminated, truncated, info = environment.step(action)
                observations[index] = observation
                history['states'].append(environment.states.copy())
                history['observations'].append(observation.copy())
                history['raw_observations'].append(np.asarray(info['raw_observation']).copy())
                history['actions'].append(np.asarray(action).copy())
                history['rewards'].append(float(reward))
                history['applied_accelerations'].append(np.asarray(info['applied_accelerations']).copy())
                history['realized_accelerations'].append(np.asarray(info['realized_accelerations']).copy())
                history['normalization_violations'].append(np.asarray(info['normalization_violation_mask'], dtype=bool))
                if 'near_shaping' in history:
                    history['best_distances'].append(float(info['best_distance_so_far']))
                    history['near_shaping'].append(float(info['reward_components']['near_shaping']))
                if 'line_cost' in history:
                    history['line_distances'].append(float(info['line_segment_distance']))
                    history['line_cost'].append(float(info['reward_components']['line_cost']))
                if 'near_shaping_uncorrected' in history:
                    history['near_shaping_uncorrected'].append(float(info['near_shaping_uncorrected']))
                    history['near_shaping_removed'].append(float(info['near_shaping_removed']))
                    history['near_counterfactual_distances'].append(float(info['near_counterfactual_distance']))
                if not (terminated or truncated):
                    next_active.append(index)
                    continue
                states = np.asarray(history['states'])
                distance = np.linalg.norm(states[:, 0, :3] - states[:, 1, :3], axis=-1)
                captured = bool(info['captured'])
                failed = bool(info['enemy_attack_success'])
                relative = 'trajectories/episode_{:03d}.npz'.format(index)
                row = dict(scenario_id=index, seed=int(seed + index), angle_degrees=float(angles[index]),
                           captured=captured, defense_failure=failed, timeout=bool(info['timeout']),
                           episode_return=float(np.sum(history['rewards'])), episode_length=environment.step_count,
                           event_time=environment.step_count * environment.dt,
                           capture_time=environment.step_count * environment.dt if captured else None,
                           min_distance=float(distance.min()), final_distance=float(distance[-1]),
                           final_attacker_base_distance=float(np.linalg.norm(states[-1, 1, :3] - environment.config['mission']['base_position'])),
                           boundary_clip_count=int(info['boundary_clip_count']), trajectory=relative)
                if 'near_shaping' in history or 'line_cost' in history:
                    from utils.formal_ppo_training import shaping_episode_metrics
                    row.update(shaping_episode_metrics(info))
                arrays = {key: np.asarray(value) for key, value in history.items()}
                for value in arrays.values():
                    if not np.isfinite(value).all():
                        raise FloatingPointError('Non-finite saved evaluation trajectory')
                np.savez_compressed(directory / relative, **arrays,
                    time=np.arange(len(states)) * environment.dt,
                    metadata_json=np.asarray(json.dumps(row, allow_nan=False)))
                rows[index] = row
            active = next_active
    write_csv(directory / 'episodes.csv', rows)
    summary = summarize(rows)
    summary.update(angles_sha256=scenario_hash(angles), deterministic=True,
                   environment_seed_base=int(seed), normalization=deepcopy(env_config['normalization']))
    write_json(directory / 'summary.json', summary)
    return summary


def angle_bin_statistics(rows, edges):
    """Assign every scenario once: [low, high), final bin includes its high edge."""
    angles = np.asarray([float(row['angle_degrees']) for row in rows])
    result = []
    total = 0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (angles >= lower) & (angles <= upper if index == len(edges) - 2 else angles < upper)
        subset = [row for row, keep in zip(rows, selected) if keep]
        total += len(subset)
        outcomes = [str(row['captured']).lower() in ('true', '1') for row in subset]
        result.append(dict(lower=float(lower), upper=float(upper), episodes=len(subset),
                           capture_rate=float(np.mean(outcomes)) if subset else None))
    if total != len(rows):
        raise ValueError('Every test scene must belong to exactly one angle bin')
    return result


def across_run_statistics(summaries):
    """Run-level sample SD and Student-t(4) interval, conditional on shared scenes."""
    if len(summaries) != 5:
        raise ValueError('The formal aggregate report requires all five completed runs')
    critical_t_df4 = 2.7764451051977987
    result = {}
    for key in ('capture_rate', 'defense_failure_rate', 'average_min_distance', 'average_return', 'average_interception_time'):
        values = [float(row[key]) for row in summaries if row[key] is not None]
        if len(values) != 5:
            result[key] = dict(available_runs=len(values), mean=float(np.mean(values)) if values else None,
                               sample_std=None, mean_ci95=None)
            continue
        mean, std = float(np.mean(values)), float(np.std(values, ddof=1))
        half = critical_t_df4 * std / np.sqrt(5)
        interval = [mean - half, mean + half]
        if key.endswith('_rate'):
            interval = [max(0, interval[0]), min(1, interval[1])]
        result[key] = dict(available_runs=5, mean=mean, sample_std=std, mean_ci95=interval)
    return result
