"""Recovery orchestration reuses completed budgets and restores saved progress."""
import json
from pathlib import Path

import torch

from scripts.resume_formal_500k import finish_seed, latest_checkpoint


def summary(seed=1001):
    return dict(status='completed', env_steps=500000, seed=seed)


def write_summary(directory):
    directory.mkdir(parents=True)
    (directory / 'summary.json').write_text(json.dumps(summary()))
    (directory / 'checkpoints').mkdir()
    (directory / 'checkpoints/final.pt').write_bytes(b'Unit orchestration fixture')


def fail_training(*args, **kwargs):
    raise AssertionError('Completed training must be reused')


def test_finished_training_and_report_are_reused(tmp_path):
    run = tmp_path / 'seed_1001'
    write_summary(run)
    post = run / 'postprocessing_001'
    post.mkdir()
    (post / 'postprocessing_manifest.json').write_text(json.dumps(dict(status='completed', output_dir=str(post))))
    actual, metrics, report = finish_seed({'training': {'total_env_steps': 500000}}, tmp_path,
        {'seed': 1001}, fail_training, fail_training)
    assert actual == run and metrics['env_steps'] == 500000
    assert report['output_dir'] == str(post)


def test_incomplete_report_is_regenerated_without_training_or_overwrite(tmp_path):
    run = tmp_path / 'seed_1001'
    write_summary(run)
    partial = run / 'postprocessing_001'
    partial.mkdir()
    (partial / 'incomplete.txt').write_text('Preserved partial report')

    def postprocess(directory, config, metrics):
        assert directory == run and metrics['env_steps'] == 500000
        target = directory / 'postprocessing_002'
        target.mkdir()
        return dict(status='completed', output_dir=str(target))

    _, _, report = finish_seed({'training': {'total_env_steps': 500000}}, tmp_path,
        {'seed': 1001}, fail_training, postprocess)
    assert Path(report['output_dir']).name == 'postprocessing_002'
    assert (partial / 'incomplete.txt').read_text() == 'Preserved partial report'


def test_partial_run_restores_largest_actual_counter_into_new_directory(tmp_path):
    seed_dir = tmp_path / 'seed_1001'
    checkpoints = seed_dir / 'checkpoints'
    checkpoints.mkdir(parents=True)
    for name, step in [('checkpoint_999999.pt', 300000), ('evaluated_320000.pt', 320000)]:
        torch.save(dict(kind='formal_single_ppo', seed=1001,
                        config={'training': {'total_env_steps': 500000}},
                        training_state={'env_steps': step}), checkpoints / name)
    expected = checkpoints / 'evaluated_320000.pt'
    before = expected.read_bytes()
    assert latest_checkpoint(seed_dir, 1001, 500000) == expected

    def train(config, output, seed, **kwargs):
        assert kwargs['resume_checkpoint'] == expected
        assert 'allow_budget_extension' not in kwargs
        assert not output.exists()
        output.mkdir()
        return summary(seed)

    directory, _, _ = finish_seed({'training': {'total_env_steps': 500000}}, tmp_path,
        {'seed': 1001}, train, lambda run, *args: dict(output_dir=str(run / 'postprocessing_001')))
    assert directory == seed_dir / 'recovery_001'
    assert expected.read_bytes() == before


def test_unstarted_seed_inherits_its_own_original_final(tmp_path):
    source = {'seed': 1001, 'checkpoint': tmp_path / 'source_final.pt'}
    source['checkpoint'].write_bytes(b'Original source fixture')

    def train(config, output, seed, **kwargs):
        assert kwargs == dict(resume_checkpoint=source['checkpoint'], allow_budget_extension=True)
        assert seed == 1001 and not output.exists()
        output.mkdir()
        return summary(seed)

    directory, _, _ = finish_seed({'training': {'total_env_steps': 500000}}, tmp_path,
        source, train, lambda run, *args: dict(output_dir=str(run / 'postprocessing_001')))
    assert directory.name == 'seed_1001'
    assert source['checkpoint'].read_bytes() == b'Original source fixture'
