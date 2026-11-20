"""Short engineering substitutes verify formal suite orchestration, without training."""
from copy import deepcopy
import json

import numpy as np
import pytest

from scripts.run_formal_experiment import run_acceptance_smoke, run_experiment
from utils.formal_ppo_config import FormalConfigurationBlocked, load_formal_config
from utils.formal_ppo_evaluation import write_csv, write_json


def fixture_config():
    config = load_formal_config()
    config['experiment']['engineering_fixture'] = True
    config['environment_config']['normalization'].update(
        confirmed=True, scales=[1.] * 16, acceleration_kind='applied',
        confirmation_source='synthetic orchestration fixture only')
    return config


def test_missing_confirmed_inputs_blocks_before_creating_outputs(tmp_path):
    config = fixture_config()
    config['environment_config']['normalization']['confirmed'] = False
    called = []
    with pytest.raises(FormalConfigurationBlocked):
        run_experiment(config, tmp_path / 'blocked', train_fn=lambda *a, **k: called.append(1))
    assert called == []
    assert not (tmp_path / 'blocked').exists()


def test_five_fresh_runs_finish_before_shared_final_test(tmp_path):
    config = fixture_config()
    untouched = deepcopy(config)
    calls, tested_angles, trained_configs = [], [], []

    def postprocess(*args):
        calls.append(('postprocess', args[2]['seed']))

    def train_stub(actual_config, output, seed, **kwargs):
        assert 'resume_checkpoint' not in kwargs
        assert actual_config == untouched
        trained_configs.append(deepcopy(actual_config))
        calls.append(('train', seed))
        output.mkdir()
        result = dict(status='completed', env_steps=300000, seed=seed)
        kwargs['post_training_fn'](output, actual_config, result)
        return result

    def load_stub(output, actual_config):
        assert len([entry for entry in calls if entry[0] == 'train']) == 5
        seed = int(output.name.split('_')[-1])
        return seed, 100000

    def test_stub(agent, environment, angles, seed, output):
        assert seed == config['experiment']['test_seed']
        assert len(angles) == 500
        calls.append(('test', agent))
        tested_angles.append(np.asarray(angles).copy())
        output.mkdir()
        write_csv(output / 'episodes.csv', [dict(angle_degrees=angle, captured=index < 400)
                                           for index, angle in enumerate(angles)])
        metrics = dict(episodes=500, captures=400, capture_rate=.8, defense_failure_rate=.2,
                       timeout_rate=0., average_interception_time=20., average_min_distance=4.,
                       average_return=1., average_episode_length=200., capture_rate_wilson_95=[.76, .83])
        write_json(output / 'summary.json', metrics)
        return metrics

    result = run_experiment(config, tmp_path / 'suite', train_fn=train_stub,
                            evaluation_fn=test_stub, post_training_fn=postprocess,
                            load_best_fn=load_stub)
    assert [seed for kind, seed in calls if kind == 'train'] == [1001, 1002, 1003, 1004, 1005]
    assert [kind for kind, seed in calls] == ['train', 'postprocess'] * 5 + ['test'] * 5
    for angles in tested_angles[1:]:
        np.testing.assert_array_equal(angles, tested_angles[0])
    assert config == untouched
    assert len(trained_configs) == 5
    assert result['passed'] is True
    assert result['total_training_steps'] == 1500000
    assert result['statistics']['capture_rate']['sample_std'] == 0
    manifest = json.loads((tmp_path / 'suite/scenarios/scenario_manifest.json').read_text())
    assert result['test_angles_sha256'] == manifest['test_sha256']
    assert (tmp_path / 'suite/FORMAL_EXPERIMENT_REPORT.md').is_file()
    assert (tmp_path / 'suite/comparison_plots/held_out_capture_rates.png').is_file()
    with pytest.raises(FileExistsError):
        run_experiment(config, tmp_path / 'suite', train_fn=train_stub, post_training_fn=postprocess)


def test_partial_training_never_produces_final_test_or_aggregate(tmp_path):
    config = fixture_config()
    calls = []

    def incomplete(config, output, seed, **kwargs):
        calls.append(seed)
        return dict(status='paused', env_steps=10000)

    def no_test(*args, **kwargs):
        raise AssertionError('Final test ran before all training completed')

    with pytest.raises(RuntimeError, match='exact training budget'):
        run_experiment(config, tmp_path / 'incomplete', train_fn=incomplete,
                        evaluation_fn=no_test, post_training_fn=lambda *args: None)
    assert calls == [1001]
    assert not (tmp_path / 'incomplete/aggregate_test_results.json').exists()


def test_smoke_uses_real_training_and_verifies_mid_rollout_recovery(tmp_path):
    config = fixture_config()
    config['environment_config']['environment']['max_episode_steps'] = 2
    config['training'].update(total_env_steps=64, num_envs=8, rollout_steps=4,
                             evaluation_interval_steps=16, validation_episodes=3)
    config['acceptance'].update(smoke_pause_steps=24, smoke_stop_steps=48)
    result = run_acceptance_smoke(config, tmp_path / 'smoke')
    assert result['passed'] and result['unfinished_rollout_samples'] == 24
    assert result['formal_training_started'] is False
    summary = json.loads((tmp_path / 'smoke/restored/summary.json').read_text())
    assert summary['status'] == 'paused' and summary['env_steps'] == 48
    assert (tmp_path / 'smoke/SMOKE_RESULT.json').is_file()
