"""Recover a sequential formal experiment without repeating completed training."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import traceback

import numpy as np
import torch
import yaml

from scripts.run_formal_500k import _hash_file, check_sources
from scripts.run_formal_experiment import (
    _aggregate_figures, _load_best, _read_rows, _snapshot_sources, _write_suite_report,
)
from utils.formal_ppo_config import load_formal_config
from utils.formal_ppo_evaluation import (
    across_run_statistics, angle_bin_statistics, evaluate_angles, scenario_hash, write_csv, write_json,
)
from utils.formal_ppo_reporting import postprocess_run
from utils.formal_ppo_training import train


def new_directory(parent, prefix):
    index = 1
    while True:
        path = Path(parent) / '{}_{:03d}'.format(prefix, index)
        try:
            path.mkdir(parents=True, exist_ok=False)
            return path
        except FileExistsError:
            index += 1


def run_directories(seed_dir):
    return [seed_dir] + sorted(seed_dir.glob('recovery_*'))


def completed_run(seed_dir, seed, budget):
    for directory in reversed(run_directories(seed_dir)):
        path = directory / 'summary.json'
        if path.exists() and (directory / 'checkpoints/final.pt').exists():
            summary = json.loads(path.read_text(encoding='utf-8'))
            if summary.get('status') == 'completed' and summary.get('env_steps') == budget and summary.get('seed') == seed:
                return directory, summary
    return None


def latest_checkpoint(seed_dir, seed, budget):
    candidates = []
    for directory in run_directories(seed_dir):
        for path in (directory / 'checkpoints').glob('*.pt'):
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            if (checkpoint.get('kind') == 'formal_single_ppo' and checkpoint.get('seed') == seed
                    and checkpoint['config']['training']['total_env_steps'] == budget):
                candidates.append((checkpoint['training_state']['env_steps'], path.stat().st_mtime_ns, path))
    if not candidates:
        raise ValueError('Interrupted run has no complete checkpoint to restore: ' + str(seed_dir))
    return max(candidates)[2]


def finish_seed(config, root, source, train_fn, postprocess_fn):
    seed, budget = source['seed'], config['training']['total_env_steps']
    seed_dir = root / 'seed_{}'.format(seed)
    completed = completed_run(seed_dir, seed, budget)
    if completed is not None:
        directory, summary = completed
        print('recovery seed={} reusing_completed_steps={}'.format(seed, budget), flush=True)
    elif not seed_dir.exists():
        directory = seed_dir
        summary = train_fn(config, directory, seed, resume_checkpoint=source['checkpoint'], allow_budget_extension=True)
    else:
        checkpoint = latest_checkpoint(seed_dir, seed, budget)
        # The trainer creates the destination exclusively.
        temporary = new_directory(seed_dir, 'recovery')
        temporary.rmdir()
        directory = temporary
        print('recovery seed={} restoring_checkpoint={}'.format(seed, checkpoint), flush=True)
        summary = train_fn(config, directory, seed, resume_checkpoint=checkpoint)
    if summary['status'] != 'completed' or summary['env_steps'] != budget:
        raise RuntimeError('Recovered run did not finish its exact budget')
    reports = sorted(directory.glob('postprocessing_*/postprocessing_manifest.json'))
    if reports:
        postprocessing = json.loads(reports[-1].read_text(encoding='utf-8'))
    else:
        postprocessing = postprocess_fn(directory, config, summary)
    return directory, summary, postprocessing


def recover_experiment(config, source_experiment, root, control, train_fn=train,
                        postprocess_fn=postprocess_run, evaluation_fn=evaluate_angles):
    root, control = Path(root).resolve(), Path(control).resolve()
    sources = check_sources(config, source_experiment)
    recorded = yaml.safe_load((root / 'experiment_config.yaml').read_text(encoding='utf-8'))
    if recorded != config:
        raise ValueError('Recovery configuration differs from the original experiment')
    recorded_sources = json.loads((root / 'continuation_sources.json').read_text(encoding='utf-8'))
    if recorded_sources != sources:
        raise ValueError('Recovery source checkpoints or hashes differ')
    manifest = json.loads((root / 'scenarios/scenario_manifest.json').read_text(encoding='utf-8'))
    if scenario_hash(manifest['test_angles']) != manifest['test_sha256']:
        raise ValueError('Frozen final-test scenario hash differs')
    _snapshot_sources(control, config)
    recovery_script = Path(__file__).resolve()
    shutil.copyfile(recovery_script, control / recovery_script.name)
    write_json(control / 'recovery_provenance.json', dict(
        previous_process='No live Python process or normal completion marker was found',
        strategy='Reuse completed models, restore unfinished runs, write new postprocessing directories',
        script_sha256=_hash_file(recovery_script), source_experiment=str(Path(source_experiment).resolve()),
        sources=sources))
    resolved, summaries = {}, []
    for source in sources:
        directory, summary, postprocessing = finish_seed(config, root, source, train_fn, postprocess_fn)
        resolved[source['seed']] = directory
        summaries.append(summary)
        write_json(control / 'seed_{}_completed.json'.format(source['seed']),
                   dict(run=str(directory), summary=summary, postprocessing=postprocessing['output_dir']))
    rows, bins = [], []
    angles = manifest['test_angles']
    for source in sources:
        seed, directory = source['seed'], resolved[source['seed']]
        agent, step = _load_best(directory, config)
        best_hash = _hash_file(directory / 'checkpoints/best.pt')
        expected = dict(best_sha256=best_hash, angles_sha256=manifest['test_sha256'])
        evaluation = directory / 'final_test'
        if evaluation.exists():
            provenance = evaluation / 'test_provenance.json'
            if (provenance.exists() and (evaluation / 'summary.json').exists()
                    and json.loads(provenance.read_text()) == expected):
                metrics = json.loads((evaluation / 'summary.json').read_text())
            else:
                evaluation = new_directory(directory, 'final_test_recovery')
                evaluation.rmdir()
                metrics = evaluation_fn(agent, config['environment_config'], angles,
                                         config['experiment']['test_seed'], evaluation)
                write_json(evaluation / 'test_provenance.json', expected)
        else:
            metrics = evaluation_fn(agent, config['environment_config'], angles,
                                     config['experiment']['test_seed'], evaluation)
            write_json(evaluation / 'test_provenance.json', expected)
        if metrics['episodes'] != len(angles) or metrics['angles_sha256'] != manifest['test_sha256']:
            raise ValueError('Recovered final test has an incomplete or different scene set')
        metrics.update(seed=seed, best_checkpoint_step=step, evaluation_directory=str(evaluation))
        rows.append(metrics)
        for row in angle_bin_statistics(_read_rows(evaluation / 'episodes.csv'), config['experiment']['angle_bins_degrees']):
            row['seed'] = seed
            bins.append(row)
    aggregate = dict(statistics=across_run_statistics(rows), seeds=config['experiment']['seeds'],
                     test_angles_sha256=manifest['test_sha256'], test_episodes_per_model=len(angles),
                     total_training_steps=sum(summary['env_steps'] for summary in summaries),
                     passed=float(np.mean([row['capture_rate'] for row in rows])) >= .8,
                     confidence_interval='Student-t(4) across five training runs, conditional on shared test set')
    result_names = ('training_summaries.json', 'test_results_by_seed.csv', 'test_angle_bins.csv',
                    'aggregate_test_results.json', 'FORMAL_EXPERIMENT_REPORT.md', 'comparison_plots')
    results = root
    if any((root / name).exists() for name in result_names):
        results = control / 'results'
        results.mkdir()
    write_json(results / 'training_summaries.json', summaries)
    write_csv(results / 'test_results_by_seed.csv', rows)
    write_csv(results / 'test_angle_bins.csv', bins)
    write_json(results / 'aggregate_test_results.json', aggregate)
    _aggregate_figures(results, rows, bins)
    description = ('五个Seed分别完整继承自身300k final至累计500k，300k后维持原终值LR。'
                   '执行中断后的恢复复用已完成训练模型并保留全部旧产物。')
    _write_suite_report(results, config, rows, aggregate, manifest, description)
    for source in sources:
        if _hash_file(source['checkpoint']) != source['sha256']:
            raise RuntimeError('Original 300k checkpoint changed')
    write_json(control / 'continuation_verification.json', dict(
        all_source_checkpoints_unchanged=True, total_additional_training_steps=sum(row['additional_steps'] for row in sources),
        total_cumulative_training_steps=aggregate['total_training_steps'],
        resolved_runs={str(seed): str(path) for seed, path in resolved.items()}, results=str(results),
        lr_decay_steps=config['training']['learning_rate_decay_steps'],
        actor_plateau_learning_rate=config['ppo']['actor_final_learning_rate'],
        critic_plateau_learning_rate=config['ppo']['critic_final_learning_rate']))
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/ppo_formal_v1_500k.yaml'))
    parser.add_argument('--source-experiment', type=Path, required=True)
    args = parser.parse_args()
    config = load_formal_config(args.config)
    root = Path(config['experiment']['output_root']).resolve()
    control = new_directory(root, 'execution_recovery')
    try:
        result = recover_experiment(config, args.source_experiment, root, control)
    except BaseException as error:
        write_json(control / 'process_completion.json', dict(exit_code=1, error=repr(error), traceback=traceback.format_exc()))
        raise
    write_json(control / 'process_completion.json', dict(exit_code=0, result=result))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
