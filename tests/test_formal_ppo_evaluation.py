"""Held-out scenario isolation and metrics on explicit engineering fixtures."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
import yaml

from utils.formal_ppo_config import FormalConfigurationBlocked, load_formal_config, validate_readiness
from utils.formal_ppo_evaluation import (
    across_run_statistics, angle_bin_statistics, create_scenarios, evaluate_angles,
)

ROOT = Path(__file__).resolve().parents[1]


def fixture_config():
    config = load_formal_config()
    environment = config['environment_config']
    environment['environment']['max_episode_steps'] = 2
    # Unit divisors deliberately exercise out-of-range recording. They are
    # synthetic test inputs, never production normalization defaults.
    environment['normalization'].update(confirmed=True, scales=[1.] * 16, acceleration_kind='applied',
                                        confirmation_source='synthetic unit test fixture')
    return config


class ZeroAgent:
    def __init__(self):
        self.actor, self.critic = torch.nn.Linear(16, 3), torch.nn.Linear(16, 1)

    def act(self, observations, deterministic=False):
        assert deterministic is True
        return {'actions': np.zeros((len(observations), 3), dtype=np.float32)}


def test_production_config_blocks_without_confirmation():
    config = load_formal_config()
    config['environment_config']['normalization']['confirmed'] = False
    with pytest.raises(FormalConfigurationBlocked, match='confirmed'):
        validate_readiness(config)


def test_scenario_sets_are_fixed_shared_and_independent(tmp_path):
    config = fixture_config()
    original = deepcopy(config)
    a = create_scenarios(config, config['environment_config'], tmp_path / 'a')
    b = create_scenarios(config, config['environment_config'], tmp_path / 'b')
    assert a == b
    assert len(a['validation_angles']) == 100 and len(a['test_angles']) == 500
    np.testing.assert_array_equal(a['validation_angles'], np.linspace(-30, 30, 100))
    assert not set(a['test_angles']).intersection(a['validation_angles'])
    assert len(set(a['test_angles'])) == 500
    assert config == original
    with pytest.raises(FileExistsError):
        create_scenarios(config, config['environment_config'], tmp_path / 'a')


def test_batched_evaluation_complete_finite_and_rng_preserved(tmp_path):
    config = fixture_config()
    agent = ZeroAgent()
    agent.actor.train()
    agent.critic.eval()
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    before = {k: v.clone() for k, v in agent.actor.state_dict().items()}
    summary = evaluate_angles(agent, config['environment_config'], [-30, 0, 30], 8000, tmp_path / 'eval')
    assert summary['episodes'] == 3 and summary['timeout_rate'] == 1
    assert summary['capture_rate'] == summary['defense_failure_rate'] == 0
    assert summary['average_interception_time'] is None
    assert random.getstate() == python_state
    current_numpy = np.random.get_state()
    assert current_numpy[0] == numpy_state[0]
    np.testing.assert_array_equal(current_numpy[1], numpy_state[1])
    assert current_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert agent.actor.training and not agent.critic.training
    for key, tensor in agent.actor.state_dict().items():
        assert torch.equal(tensor, before[key])
    for path in sorted((tmp_path / 'eval/trajectories').glob('*.npz')):
        with np.load(path, allow_pickle=False) as archive:
            assert archive['states'].shape == (3, 2, 6)
            assert archive['observations'].shape == (3, 16)
            assert archive['actions'].shape == (2, 3)
            assert archive['normalization_violations'].shape == (2, 16)
            assert np.isfinite(archive['states']).all()
            assert json.loads(archive['metadata_json'].item())['timeout']
    with pytest.raises(FileExistsError):
        evaluate_angles(agent, config['environment_config'], [0], 8000, tmp_path / 'eval')


def test_every_angle_boundary_assigned_once_and_csv_bools():
    angles = [-30, -20, -10, 0, 10, 20, 30]
    rows = [dict(angle_degrees=a, captured='True' if i % 2 else 'False') for i, a in enumerate(angles)]
    bins = angle_bin_statistics(rows, [-30, -20, -10, 0, 10, 20, 30])
    assert [row['episodes'] for row in bins] == [1, 1, 1, 1, 1, 2]
    assert bins[0]['capture_rate'] == 0 and bins[1]['capture_rate'] == 1
    assert bins[-1]['capture_rate'] == .5


def test_run_statistics_treat_five_runs_as_units():
    data = [dict(capture_rate=x, defense_failure_rate=1-x, average_min_distance=10,
                 average_return=1, average_interception_time=15) for x in [.7, .75, .8, .85, .9]]
    result = across_run_statistics(data)
    assert result['capture_rate']['mean'] == pytest.approx(.8)
    assert result['capture_rate']['sample_std'] == pytest.approx(np.std([.7, .75, .8, .85, .9], ddof=1))
    assert result['capture_rate']['mean_ci95'][0] < .8 < result['capture_rate']['mean_ci95'][1]
    data[-1]['average_interception_time'] = None
    assert across_run_statistics(data)['average_interception_time']['available_runs'] == 4
    with pytest.raises(ValueError):
        across_run_statistics(data[:-1])
