"""Acceptance-gated, sequential five-seed formal 1v1 PPO experiment."""

import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch
import yaml

from algorithms.ppo import PPOAgent
from utils.formal_ppo_config import FormalConfigurationBlocked, load_formal_config, validate_readiness
from utils.formal_ppo_evaluation import (
    across_run_statistics, angle_bin_statistics, create_scenarios, evaluate_angles,
    write_csv, write_json,
)
from utils.formal_ppo_training import _state_hash, train


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    'requirements.txt', 'FORMAL_TRAINING_SPEC_V1.md', 'BASELINE_SPEC.md',
    'envs/__init__.py', 'envs/interception_env.py', 'envs/dynamics.py',
    'envs/attacker_policy.py', 'envs/evader_policy.py', 'envs/pursuit_env.py', 'envs/rewards.py',
    'algorithms/__init__.py', 'algorithms/ppo/__init__.py', 'algorithms/ppo/agent.py', 'algorithms/ppo/buffer.py',
    'algorithms/mappo/__init__.py', 'algorithms/mappo/networks.py', 'algorithms/mappo/mappo.py',
    'algorithms/mappo/buffer.py', 'algorithms/mappo/normalization.py', 'utils/__init__.py',
    'utils/experiment.py', 'utils/formal_ppo_config.py',
    'utils/formal_ppo_training.py', 'utils/formal_ppo_evaluation.py', 'utils/formal_ppo_reporting.py',
    'scripts/__init__.py', 'scripts/train_formal_ppo.py', 'scripts/run_formal_experiment.py',
    'scripts/run_formal_500k.py',
)


def _snapshot_sources(output, config):
    destination = output / 'source_snapshot'
    destination.mkdir()
    rows = []
    for relative in SOURCE_FILES:
        source, copied = ROOT / relative, destination / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, copied)
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if hashlib.sha256(copied.read_bytes()).hexdigest() != source_hash:
            raise RuntimeError('Source snapshot failed its hash check')
        rows.append(dict(path=relative, sha256=source_hash))
    for key, name in (('_config_source', 'training_source.yaml'), ('_environment_source', 'environment_source.yaml')):
        if config.get(key):
            source = Path(config[key])
            shutil.copyfile(source, destination / name)
            rows.append(dict(path=name, original_path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    write_json(output / 'source_manifest.json', dict(files=rows,
        resolved_experiment_config='experiment_config.yaml',
        note='Run metadata records Git HEAD and dirty status; these files preserve the actual uncommitted implementation'))


def _read_rows(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def run_acceptance_smoke(config, output_dir, expected_seed_count=5):
    """Exercise production settings and exact mid-rollout recovery before training.

    The original 300k scheduler horizon is retained. These short engineering
    runs have their own directory and never count toward the five formal runs.
    """
    validate_readiness(config, expected_seed_count=expected_seed_count)
    pause = config['acceptance']['smoke_pause_steps']
    stop = config['acceptance']['smoke_stop_steps']
    count = config['training']['num_envs']
    if not (0 < pause < stop <= config['training']['total_env_steps']) or pause % count or stop % count:
        raise ValueError('Smoke boundaries must be ordered, complete vector steps')
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    seed = config['experiment']['seeds'][0]
    expected_dir, paused_dir, restored_dir = (output / name for name in ('uninterrupted', 'interrupted', 'restored'))
    train(config, expected_dir, seed, stop_at_steps=stop)
    paused = train(config, paused_dir, seed, stop_at_steps=pause)
    if paused['pending_rollout_samples'] == 0:
        raise ValueError('Smoke pause must retain an unfinished rollout')
    checkpoint_path = paused_dir / 'checkpoints/paused_{:06d}.pt'.format(pause)
    train(config, restored_dir, seed, resume_checkpoint=checkpoint_path, stop_at_steps=stop)
    checkpoints = [torch.load(directory / 'checkpoints/paused_{:06d}.pt'.format(stop),
                               map_location='cpu', weights_only=False)
                   for directory in (expected_dir, restored_dir)]
    for checkpoint in checkpoints:
        checkpoint['agent']['config'].pop('output_dir', None)
    for name in ('agent', 'training_state', 'rng_state'):
        if _state_hash(checkpoints[0][name]) != _state_hash(checkpoints[1][name]):
            raise RuntimeError('Smoke checkpoint recovery diverged in ' + name)
    for csv_path in expected_dir.glob('*.csv'):
        if csv_path.read_bytes() != (restored_dir / csv_path.name).read_bytes():
            raise RuntimeError('Smoke checkpoint recovery changed ' + csv_path.name)
    summary = dict(passed=True, production_settings=True, formal_training_started=False,
                   seed=seed, pause_steps=pause, stop_steps=stop,
                   unfinished_rollout_samples=paused['pending_rollout_samples'],
                   recovery='Bitwise model, Adam, scheduler, environments, buffer, RNG and CSV equality')
    write_json(output / 'SMOKE_RESULT.json', summary)
    return summary


def _load_best(run_dir, config):
    path = Path(run_dir) / 'checkpoints/best.pt'
    checkpoint = torch.load(path, map_location=config.get('device', 'cpu'), weights_only=False)
    if checkpoint.get('kind') != 'formal_single_ppo' or not checkpoint.get('is_best_checkpoint'):
        raise ValueError('Expected the run\'s validation-selected formal best checkpoint')
    agent = PPOAgent(checkpoint['config'])
    agent.load_state_dict(checkpoint['agent'])
    return agent, int(checkpoint['training_state']['env_steps'])


def _aggregate_figures(output, rows, bins):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    directory = output / 'comparison_plots'
    directory.mkdir()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar([str(row['seed']) for row in rows], [row['capture_rate'] for row in rows])
    axes[0].axhline(.8, color='black', linestyle='--', label='Target 80%')
    axes[0].set(xlabel='Independent training seed', ylabel='Held-out capture rate', ylim=(0, 1))
    axes[0].legend()
    for seed in sorted({row['seed'] for row in bins}):
        selected = [row for row in bins if row['seed'] == seed]
        axes[1].plot([(row['lower'] + row['upper']) / 2 for row in selected],
                     [np.nan if row['capture_rate'] is None else row['capture_rate'] for row in selected],
                     marker='o', label=str(seed))
    axes[1].set(xlabel='Attacker angle bin center (deg)', ylabel='Held-out capture rate', ylim=(0, 1))
    axes[1].legend(title='Seed')
    figure.tight_layout()
    figure.savefig(directory / 'held_out_capture_rates.png', dpi=180)
    plt.close(figure)


def _write_suite_report(output, config, rows, aggregate, manifest, initialization_description=None):
    rate = aggregate['statistics']['capture_rate']
    lines = [
        '# 单机 PPO 正式训练综合报告', '',
        initialization_description or '五次训练从零初始化，按 seed 顺序执行，每次 {:,} aggregate environment steps。'.format(config['training']['total_env_steps']),
        'Best 由每 10k 步的固定 100 场景验证选择，最终测试在五次训练全部结束后运行。', '',
        '| Seed | Best step | 测试捕获率 | 防御失败率 | 超时率 | 成功拦截时间 s | 平均最小距离 m |',
        '| --- | --- | --- | --- | --- | --- | --- |',
    ]
    if config['experiment'].get('engineering_fixture'):
        lines[2:2] = ['**工程测试 fixture，本文用于检验报告生成，数值未用于正式模型性能结论。**', '']
    for row in rows:
        capture_time = row['average_interception_time']
        time_text = '无成功回合' if capture_time is None else '{:.3f}'.format(capture_time)
        lines.append('| {seed} | {best_checkpoint_step} | {capture_rate:.1%} | {defense_failure_rate:.1%} | '
                     '{timeout_rate:.1%} | {time} | {average_min_distance:.3f} |'.format(time=time_text, **row))
    lines.extend([
        '', '五次测试捕获率均值 {:.2%}，跨训练 seed 样本标准差 {:.2%}。'.format(rate['mean'], rate['sample_std']),
        '均值的 95% Student-t 区间为 [{:.2%}, {:.2%}]，以五次独立训练为统计单位。'.format(*rate['mean_ci95']),
        '该区间以本次共享的 500 个测试场景为条件，每个模型另记录 Wilson 二项区间。',
        '', '整体达标状态为 {}，判定条件为五模型平均测试捕获率 ≥80%。'.format('达到' if aggregate['passed'] else '未达到'),
        '', '## 可复现记录', '',
        '训练 seeds 为 {}，验证场景 seed 为 {}，测试场景 seed 为 {}。'.format(
            config['experiment']['seeds'], config['experiment']['validation_seed'], config['experiment']['test_seed']),
        '验证场景 SHA256 为 `{}`。'.format(manifest['validation_sha256']),
        '测试场景 SHA256 为 `{}`。'.format(manifest['test_sha256']),
        '验证角度包含区间端点，最终测试角度独立均匀采样，共享于全部模型。',
        '', '## 文件入口', '',
        '- `experiment_config.yaml` 保存本轮完整解析配置。',
        '- `scenarios/` 保存验证集和最终测试集的角度、环境 seed 及哈希。',
        '- `seed_*/` 保存每次训练的配置、原始 CSV、checkpoint、曲线、演化图和 best/final 视频。',
        '- `seed_*/final_test/` 保存冻结 best 的 500 回合结果和完整轨迹。',
        '- `test_results_by_seed.csv` 与 `test_angle_bins.csv` 保存逐 seed 和分区统计。',
        '- `aggregate_test_results.json` 保存均值、标准差、区间及达标判定。',
        '- `comparison_plots/held_out_capture_rates.png` 保存模型与角度分区比较。', '',
        '学习率、早停、环境终止和归一化的具体设置以完整配置和各 run 报告为准。',
    ])
    (output / 'FORMAL_EXPERIMENT_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run_experiment(config, output_dir=None, train_fn=None, evaluation_fn=None,
                   post_training_fn=None, load_best_fn=None, initialization_description=None):
    """Run all fresh seeds before any final test, without reusing old outputs.

    Optional callables let acceptance tests exercise ordering with short,
    explicitly marked fixtures. The command line always uses real modules.
    """
    config = deepcopy(config)
    validate_readiness(config)
    if train_fn is None:
        train_fn = train
    if evaluation_fn is None:
        evaluation_fn = evaluate_angles
    if load_best_fn is None:
        load_best_fn = _load_best
    if post_training_fn is None:
        from utils.formal_ppo_reporting import postprocess_run
        post_training_fn = postprocess_run
    output = Path(output_dir or config['experiment']['output_root']).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'experiment_config.yaml').open('x', encoding='utf-8') as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
    _snapshot_sources(output, config)
    manifest = create_scenarios(config, config['environment_config'], output / 'scenarios')
    fixture_notice = ('ENGINEERING FIXTURE ONLY. These results test orchestration and reporting.\n\n'
                      if config['experiment'].get('engineering_fixture') else '')
    (output / 'README.md').write_text(
        '# Formal PPO v1\n\n'
        + fixture_notice +
        'Five independent training runs execute sequentially. Final testing begins after every run completes.\n\n'
        + (initialization_description + '\n\n' if initialization_description else '') +
        'See experiment_config.yaml, scenarios/, seed_1001/ through seed_1005/, '
        'and the completed FORMAL_EXPERIMENT_REPORT.md.\n', encoding='utf-8')
    summaries = []
    for seed in config['experiment']['seeds']:
        summary = train_fn(config, output / 'seed_{}'.format(seed), seed,
                           post_training_fn=post_training_fn)
        if summary.get('status') != 'completed' or summary['env_steps'] != config['training']['total_env_steps']:
            raise RuntimeError('A run did not complete its exact training budget; final testing is blocked')
        summaries.append(summary)
    write_json(output / 'training_summaries.json', summaries)
    rows, bins = [], []
    angles = manifest['test_angles']
    for seed in config['experiment']['seeds']:
        run_dir = output / 'seed_{}'.format(seed)
        agent, best_step = load_best_fn(run_dir, config)
        metrics = evaluation_fn(agent, config['environment_config'], angles,
                                config['experiment']['test_seed'], run_dir / 'final_test')
        metrics.update(seed=seed, best_checkpoint_step=best_step)
        rows.append(metrics)
        for row in angle_bin_statistics(_read_rows(run_dir / 'final_test/episodes.csv'),
                                         config['experiment']['angle_bins_degrees']):
            row['seed'] = seed
            bins.append(row)
    aggregate = dict(statistics=across_run_statistics(rows), seeds=config['experiment']['seeds'],
                     test_angles_sha256=manifest['test_sha256'], test_episodes_per_model=len(angles),
                     total_training_steps=sum(row['env_steps'] for row in summaries),
                     passed=float(np.mean([row['capture_rate'] for row in rows])) >= .8,
                     confidence_interval='Student-t(4) across five training runs, conditional on shared test set')
    write_csv(output / 'test_results_by_seed.csv', rows)
    write_csv(output / 'test_angle_bins.csv', bins)
    write_json(output / 'aggregate_test_results.json', aggregate)
    _aggregate_figures(output, rows, bins)
    _write_suite_report(output, config, rows, aggregate, manifest, initialization_description)
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/ppo_formal_v1.yaml'))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--check-config', action='store_true')
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--smoke-output', type=Path)
    args = parser.parse_args()
    config = load_formal_config(args.config)
    try:
        validate_readiness(config)
    except FormalConfigurationBlocked as error:
        print(json.dumps(dict(ready=False, blocker=str(error), training_started=False), ensure_ascii=False, indent=2))
        raise SystemExit(2)
    if args.check_config:
        print(json.dumps(dict(ready=True, seeds=config['experiment']['seeds'],
                              aggregate_steps_per_run=config['training']['total_env_steps'],
                              num_envs=config['training']['num_envs']), indent=2))
        return
    output = Path(args.output_dir or config['experiment']['output_root'])
    if output.exists():
        parser.error('Use a new formal experiment directory')
    smoke_output = args.smoke_output or output.parent / (output.name + '_acceptance_smoke')
    smoke_result = run_acceptance_smoke(config, smoke_output)
    print(json.dumps(smoke_result, ensure_ascii=False, indent=2))
    if args.smoke_only:
        return
    print(json.dumps(run_experiment(config, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
