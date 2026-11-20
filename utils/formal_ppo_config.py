"""Load the independent formal configuration and enforce readiness prerequisites."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'configs/ppo_formal_v1.yaml'


class FormalConfigurationBlocked(ValueError):
    pass


def load_formal_config(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    env_path = Path(config['env_config'])
    env_path = env_path if env_path.is_absolute() else path.parent / env_path
    environment = yaml.safe_load(env_path.read_text(encoding='utf-8'))
    result = deepcopy(config)
    result['environment_config'] = environment
    result['_config_source'] = str(path)
    result['_environment_source'] = str(env_path.resolve())
    return result


def validate_readiness(config, expected_seed_count=5):
    environment = config['environment_config']
    normalization = environment.get('normalization', {})
    if normalization.get('confirmed') is not True:
        raise FormalConfigurationBlocked(
            'User-confirmed 16D physical scaling divisors and acceleration definition are missing. '
            'Provide normalization.scales, normalization.acceleration_kind, and confirmation_source before formal acceptance.')
    scale = np.asarray(normalization.get('scales'), dtype=float)
    if scale.shape != (16,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise FormalConfigurationBlocked('normalization.scales must contain 16 confirmed finite positive divisors')
    if normalization.get('acceleration_kind') != 'applied':
        raise FormalConfigurationBlocked('Formal observation acceleration_kind must be post-jerk applied acceleration')
    training = config['training']
    for key in ('total_env_steps', 'num_envs', 'rollout_steps', 'evaluation_interval_steps', 'validation_episodes'):
        value = training[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError('Training setting must be a positive integer: ' + key)
    for key in ('total_env_steps', 'evaluation_interval_steps'):
        if training[key] % training['num_envs']:
            raise ValueError('Aggregate counter must align with the vector environment count: ' + key)
    experiment = config['experiment']
    seeds = experiment['seeds']
    if len(seeds) != expected_seed_count or len(set(seeds)) != expected_seed_count:
        raise ValueError('{} independent training seeds are required'.format(expected_seed_count))
    if len(set(seeds + [experiment['validation_seed'], experiment['test_seed']])) != expected_seed_count + 2:
        raise ValueError('Training, validation and test root seed domains must be distinct')
    # Construct once to validate the exact same physical/normalization contract
    # consumed by training and all held-out inference entry points.
    from envs.interception_env import InterceptionEnv
    InterceptionEnv(environment, seed=seeds[0])
    return config
