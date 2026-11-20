"""Tiny real-model fixtures exercise the cumulative-budget continuation entry."""
from copy import deepcopy
import json

import pytest

from scripts.run_formal_500k import check_sources, continue_experiment, run_extension_smoke
from utils.formal_ppo_config import load_formal_config
from utils.formal_ppo_training import train


@pytest.fixture(scope='module')
def completed_sources(tmp_path_factory):
    directory = tmp_path_factory.mktemp('extension_entry')
    config = load_formal_config()
    config['experiment']['engineering_fixture'] = True
    config['experiment']['test_episodes'] = 6
    config['environment_config']['environment']['max_episode_steps'] = 2
    config['network'].update(actor_hidden_sizes=[8, 8], critic_hidden_sizes=[8, 8])
    config['ppo'].update(update_epochs=1, minibatch_size=4)
    config['training'].update(total_env_steps=24, num_envs=2, rollout_steps=4,
                              evaluation_interval_steps=8, validation_episodes=3)
    config['acceptance'].update(smoke_pause_steps=10, smoke_stop_steps=16)
    source = directory / 'source'
    for seed in config['experiment']['seeds']:
        train(config, source / 'seed_{}'.format(seed), seed)
    target = deepcopy(config)
    target['training'].update(total_env_steps=48, learning_rate_decay_steps=24)
    target['experiment']['output_root'] = str(directory / 'target')
    return directory, source, target


def test_extension_entry_preflight_and_real_partial_buffer_smoke(completed_sources):
    directory, source, target = completed_sources
    rows = check_sources(target, source)
    assert [row['seed'] for row in rows] == [1001, 1002, 1003, 1004, 1005]
    assert all(row['source_steps'] == row['additional_steps'] == 24 for row in rows)
    result = run_extension_smoke(target, source, directory / 'smoke')
    assert result['passed'] and result['source_checkpoint_unchanged']
    assert result['pause_steps'] == 34 and result['stop_steps'] == 40
    assert result['pending_rollout_samples'] == 2
    assert result['preserved_learning_rate_decay_steps'] == 24


def test_all_five_continuations_preserve_sources_and_produce_shared_test(completed_sources):
    directory, source, target = completed_sources
    result = continue_experiment(target, source, directory / 'target', post_training_fn=lambda *args: None)
    assert result['total_training_steps'] == 48 * 5
    provenance = json.loads((directory / 'target/continuation_verification.json').read_text())
    assert provenance['all_source_checkpoints_unchanged']
    assert provenance['total_additional_training_steps'] == 24 * 5
    assert provenance['actor_plateau_learning_rate'] == 3e-5
    assert provenance['critic_plateau_learning_rate'] == 1e-4
    report = (directory / 'target/FORMAL_EXPERIMENT_REPORT.md').read_text(encoding='utf-8')
    assert '分别从各自300k final完整恢复' in report
    assert '工程测试 fixture' in report
    for seed in target['experiment']['seeds']:
        summary = json.loads((directory / 'target' / 'seed_{}'.format(seed) / 'summary.json').read_text())
        assert summary['status'] == 'completed' and summary['env_steps'] == 48
        assert summary['resumed_from'].endswith('final.pt')


def test_wrong_environment_is_rejected_before_any_training(completed_sources):
    directory, source, target = completed_sources
    changed = deepcopy(target)
    changed['environment_config']['capture']['radius'] = 6.
    with pytest.raises(ValueError, match='environment differs'):
        check_sources(changed, source)
