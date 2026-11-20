"""Controlled three-seed reward experiment configuration and orchestration."""

from copy import deepcopy
from pathlib import Path
import inspect
import json

import numpy as np
import pytest

from scripts.run_formal_experiment import run_acceptance_smoke
from utils.formal_ppo_config import load_formal_config, validate_readiness
from utils.formal_ppo_evaluation import create_scenarios


ROOT = Path(__file__).resolve().parents[1]


def configurations():
    return (
        load_formal_config(ROOT / 'configs/ppo_formal_v1.yaml'),
        load_formal_config(ROOT / 'configs/ppo_reward_ab.yaml'),
    )


def test_b_environment_only_adds_confirmed_shaping():
    baseline, treatment = configurations()
    environment = deepcopy(treatment['environment_config'])
    shaping = environment['reward'].pop('best_so_far_shaping')
    assert shaping == {'length_scale_m': 20.0, 'beta': 100.0}
    assert environment == baseline['environment_config']


def test_b_preserves_all_training_and_policy_parameters():
    baseline, treatment = configurations()
    for field in ('device', 'torch_num_threads', 'network', 'ppo', 'training', 'acceptance'):
        assert treatment[field] == baseline[field]
    assert treatment['training']['total_env_steps'] == 300000
    assert treatment['training']['num_envs'] == 8
    assert treatment['experiment']['seeds'] == [1001, 1002, 1003]
    assert treatment['experiment']['output_root'] != baseline['experiment']['output_root']
    expected = deepcopy(baseline['experiment'])
    expected.update(seeds=[1001, 1002, 1003], output_root=treatment['experiment']['output_root'])
    assert treatment['experiment'] == expected


def test_three_seed_readiness_requires_explicit_protocol_and_preserves_inputs():
    _, config = configurations()
    before = deepcopy(config)
    assert validate_readiness(config, expected_seed_count=3) == before
    assert config == before
    with pytest.raises(ValueError, match='independent training seeds'):
        validate_readiness(config)
    assert inspect.signature(validate_readiness).parameters['expected_seed_count'].default == 5
    assert inspect.signature(run_acceptance_smoke).parameters['expected_seed_count'].default == 5


@pytest.mark.parametrize('seeds', [[1001, 1001, 1003], [1001, 1002], [1001, 1002, 2001]])
def test_invalid_three_seed_protocol_is_rejected(seeds):
    _, config = configurations()
    config['experiment']['seeds'] = seeds
    with pytest.raises(ValueError):
        validate_readiness(config, expected_seed_count=3)


def test_b_reuses_exact_validation_and_test_scenarios(tmp_path):
    baseline, treatment = configurations()
    first = create_scenarios(baseline, baseline['environment_config'], tmp_path / 'A')
    second = create_scenarios(treatment, treatment['environment_config'], tmp_path / 'B')
    for field in ('validation_angles', 'test_angles'):
        np.testing.assert_array_equal(first[field], second[field])
    for field in ('validation_sha256', 'test_sha256', 'validation_seed', 'test_seed'):
        assert first[field] == second[field]
    assert len(second['validation_angles']) == 100
    assert len(second['test_angles']) == 500
    for filename in ('validation_scenarios.csv', 'test_scenarios.csv'):
        assert (tmp_path / 'A' / filename).read_bytes() == (tmp_path / 'B' / filename).read_bytes()


def test_completed_seed_reuses_training_and_postprocessing(tmp_path):
    from scripts.run_reward_ab import finish_seed

    _, config = configurations()
    run = tmp_path / 'seed_1001'
    post = run / 'postprocessing_001'
    post.mkdir(parents=True)
    summary = dict(status='completed', env_steps=300000, seed=1001)
    manifest = dict(status='completed', output_dir=str(post))
    (run / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
    (post / 'postprocessing_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')

    def should_not_run(*args, **kwargs):
        raise AssertionError('Completed artifacts were recomputed')

    actual = finish_seed(config, tmp_path, 1001, should_not_run, should_not_run)
    assert actual == (summary, manifest)


def test_completed_training_only_generates_missing_postprocessing(tmp_path):
    from scripts.run_reward_ab import finish_seed

    _, config = configurations()
    run = tmp_path / 'seed_1001'
    run.mkdir()
    summary = dict(status='completed', env_steps=300000, seed=1001)
    (run / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
    calls = []

    def should_not_train(*args, **kwargs):
        raise AssertionError('A completed seed was trained again')

    def postprocess(actual_run, actual_config, actual_summary):
        calls.append((actual_run, actual_summary))
        assert actual_config == config
        return dict(status='completed', output_dir=str(actual_run / 'postprocessing_001'))

    result = finish_seed(config, tmp_path, 1001, should_not_train, postprocess)
    assert result[0] == summary
    assert calls == [(run, summary)]


def test_partial_directory_does_not_restart_training(tmp_path):
    from scripts.run_reward_ab import finish_seed

    _, config = configurations()
    (tmp_path / 'seed_1001').mkdir()
    calls = []
    with pytest.raises(RuntimeError, match='checkpoint recovery'):
        finish_seed(config, tmp_path, 1001, lambda *args: calls.append(args))
    assert calls == []


def test_suite_orders_three_fresh_seeds_then_shared_evaluation_and_reuses_completion(tmp_path, monkeypatch):
    from scripts import run_reward_ab as runner

    _, config = configurations()
    before = deepcopy(config)
    baseline = tmp_path / 'historical_A'
    output = tmp_path / 'B'
    scenarios = create_scenarios(config, config['environment_config'], tmp_path / 'frozen_scenarios')
    calls = []

    def preflight(actual_config, actual_baseline):
        assert actual_config == before
        assert actual_baseline == baseline
        return dict(historical_source_sha256={})

    def prepare(actual_config, actual_output, actual_baseline, checked):
        actual_output.mkdir(exist_ok=True)
        return scenarios

    def train(actual_config, run, seed):
        assert actual_config == before
        calls.append(('train', seed))
        run.mkdir()
        summary = dict(status='completed', env_steps=300000, seed=seed)
        (run / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
        return summary

    def postprocess(run, actual_config, summary):
        calls.append(('postprocess', summary['seed']))
        post = run / 'postprocessing_001'
        post.mkdir()
        manifest = dict(status='completed', output_dir=str(post))
        (post / 'postprocessing_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        return manifest

    def evaluate(checkpoint, destination, angles, seed, evaluation_fn):
        assert [item[1] for item in calls if item[0] == 'train'] == [1001, 1002, 1003]
        np.testing.assert_array_equal(angles, scenarios['test_angles'])
        assert len(angles) == 500
        assert seed == 3001
        if not (destination / 'summary.json').exists():
            calls.append(('evaluate', str(checkpoint)))
            destination.mkdir(parents=True)
            (destination / 'summary.json').write_text('{}', encoding='utf-8')
        return {}

    def analyze(actual_output, actual_baseline):
        assert len([item for item in calls if item[0] == 'evaluate']) == 9
        assert actual_baseline == baseline
        report = actual_output / 'comparison' / 'REPORT.md'
        report.parent.mkdir(exist_ok=True)
        report.write_text('Engineering orchestration fixture', encoding='utf-8')
        return report

    monkeypatch.setattr(runner, 'preflight', preflight)
    monkeypatch.setattr(runner, '_prepare', prepare)
    monkeypatch.setattr(runner, 'evaluate_checkpoint', evaluate)
    first = runner.run_experiment(config, output, baseline, train_fn=train,
                                 postprocess_fn=postprocess, analysis_fn=analyze)
    assert first['training_steps'] == 900000
    assert calls[:6] == [('train', 1001), ('postprocess', 1001), ('train', 1002),
                         ('postprocess', 1002), ('train', 1003), ('postprocess', 1003)]
    evaluated = [item[1] for item in calls if item[0] == 'evaluate']
    assert sum(str(baseline) in path for path in evaluated) == 3
    assert sum(path.endswith('best.pt') for path in evaluated) == 3
    assert sum(path.endswith('final.pt') for path in evaluated) == 6
    original_calls = list(calls)
    second = runner.run_experiment(config, output, baseline, train_fn=train,
                                  postprocess_fn=postprocess, analysis_fn=analyze)
    assert second['training_steps'] == 900000
    assert calls == original_calls
    assert config == before


def test_preparation_rejects_historical_scenario_mismatch_before_snapshot(tmp_path, monkeypatch):
    from scripts import run_reward_ab as runner

    _, config = configurations()
    baseline = tmp_path / 'historical_A'
    (baseline / 'scenarios').mkdir(parents=True)
    (baseline / 'scenarios/scenario_manifest.json').write_text(
        json.dumps(dict(validation_sha256='different', test_sha256='different')), encoding='utf-8')

    def should_not_snapshot(*args):
        raise AssertionError('Snapshot created before matching scene verification')

    monkeypatch.setattr(runner, '_snapshot_sources', should_not_snapshot)
    with pytest.raises(ValueError, match='scenario sets differ'):
        runner._prepare(config, tmp_path / 'B', baseline, {})
