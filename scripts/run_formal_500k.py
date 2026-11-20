"""Continue each completed formal run from its own 300k final to 500k."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from scripts.run_formal_experiment import run_experiment
from utils.formal_ppo_config import load_formal_config, validate_readiness
from utils.formal_ppo_evaluation import write_json
from utils.formal_ppo_training import _state_hash, train


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_sources(config, source_experiment):
    validate_readiness(config)
    source = Path(source_experiment).resolve()
    horizon = config['training']['learning_rate_decay_steps']
    if horizon >= config['training']['total_env_steps']:
        raise ValueError('Continuation target must exceed the preserved decay horizon')
    rows = []
    for seed in config['experiment']['seeds']:
        run = source / 'seed_{}'.format(seed)
        summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
        path = run / 'checkpoints/final.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if (summary['status'] != 'completed' or summary['env_steps'] != horizon
                or checkpoint['seed'] != seed or checkpoint['training_state']['env_steps'] != horizon
                or checkpoint['training_state']['rollout_buffer']['size'] != 0):
            raise ValueError('Every source must be its own completed 300k final checkpoint')
        if checkpoint['environment_config'] != config['environment_config']:
            raise ValueError('Source environment differs from the confirmed continuation environment')
        if checkpoint['config']['network'] != config['network'] or checkpoint['config']['ppo'] != config['ppo']:
            raise ValueError('Continuation may not change network or PPO hyperparameters')
        rows.append(dict(seed=seed, checkpoint=str(path), sha256=_hash_file(path),
                         source_steps=horizon, target_steps=config['training']['total_env_steps'],
                         additional_steps=config['training']['total_env_steps'] - horizon))
    return rows


def run_extension_smoke(config, source_experiment, output_dir):
    sources = check_sources(config, source_experiment)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = sources[0]
    pause = source['source_steps'] + config['acceptance']['smoke_pause_steps']
    stop = source['source_steps'] + config['acceptance']['smoke_stop_steps']
    expected, interrupted, restored = (output / name for name in ('uninterrupted', 'interrupted', 'restored'))
    train(config, expected, source['seed'], resume_checkpoint=source['checkpoint'],
          allow_budget_extension=True, stop_at_steps=stop)
    paused = train(config, interrupted, source['seed'], resume_checkpoint=source['checkpoint'],
                   allow_budget_extension=True, stop_at_steps=pause)
    if not paused['pending_rollout_samples']:
        raise RuntimeError('Extension smoke did not preserve a partial rollout')
    train(config, restored, source['seed'],
          resume_checkpoint=interrupted / 'checkpoints/paused_{:06d}.pt'.format(pause), stop_at_steps=stop)
    bundles = [torch.load(run / 'checkpoints/paused_{:06d}.pt'.format(stop), map_location='cpu', weights_only=False)
               for run in (expected, restored)]
    for bundle in bundles:
        bundle['agent']['config'].pop('output_dir', None)
    for key in ('agent', 'training_state', 'rng_state'):
        if _state_hash(bundles[0][key]) != _state_hash(bundles[1][key]):
            raise RuntimeError('Extension smoke diverged in ' + key)
    for path in expected.glob('*.csv'):
        if path.read_bytes() != (restored / path.name).read_bytes():
            raise RuntimeError('Extension smoke changed ' + path.name)
    if _hash_file(source['checkpoint']) != source['sha256']:
        raise RuntimeError('Source checkpoint changed during smoke')
    result = dict(passed=True, source_steps=source['source_steps'], pause_steps=pause, stop_steps=stop,
                  pending_rollout_samples=paused['pending_rollout_samples'],
                  preserved_learning_rate_decay_steps=config['training']['learning_rate_decay_steps'],
                  resumed_model_optimizer_rng_environment_buffer_csv_bitwise_equal=True,
                  source_checkpoint_unchanged=True)
    write_json(output / 'SMOKE_RESULT.json', result)
    return result


def continue_experiment(config, source_experiment, output_dir=None, post_training_fn=None):
    sources = check_sources(config, source_experiment)
    by_seed = {row['seed']: row for row in sources}

    def train_continuation(actual_config, run, seed, **kwargs):
        provenance = Path(run).parent / 'continuation_sources.json'
        if not provenance.exists():
            write_json(provenance, sources)
        return train(actual_config, run, seed, resume_checkpoint=by_seed[seed]['checkpoint'],
                     allow_budget_extension=True, **kwargs)

    description = ('五个Seed分别从各自300k final完整恢复至累计500k，继承Actor、Critic、Adam、环境与RNG。'
                   '保留原300k线性学习率轨迹，之后固定Actor LR=3e-5、Critic LR=1e-4。')
    result = run_experiment(config, output_dir, train_fn=train_continuation,
                             post_training_fn=post_training_fn, initialization_description=description)
    for source in sources:
        if _hash_file(source['checkpoint']) != source['sha256']:
            raise RuntimeError('Original final checkpoint changed')
    output = Path(output_dir or config['experiment']['output_root']).resolve()
    write_json(output / 'continuation_verification.json', dict(
        all_source_checkpoints_unchanged=True, total_additional_training_steps=sum(row['additional_steps'] for row in sources),
        total_cumulative_training_steps=result['total_training_steps'],
        lr_decay_steps=config['training']['learning_rate_decay_steps'],
        actor_plateau_learning_rate=config['ppo']['actor_final_learning_rate'],
        critic_plateau_learning_rate=config['ppo']['critic_final_learning_rate']))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/ppo_formal_v1_500k.yaml'))
    parser.add_argument('--source-experiment', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--smoke-output', type=Path)
    parser.add_argument('--check-config', action='store_true')
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    config = load_formal_config(args.config)
    sources = check_sources(config, args.source_experiment)
    if args.check_config:
        print(json.dumps(dict(ready=True, sources=sources), ensure_ascii=False, indent=2))
        return
    output = Path(args.output_dir or config['experiment']['output_root'])
    if output.exists():
        parser.error('Use a new continuation experiment directory')
    smoke = args.smoke_output or output.parent / (output.name + '_acceptance_smoke')
    print(json.dumps(run_extension_smoke(config, args.source_experiment, smoke), ensure_ascii=False, indent=2))
    if not args.smoke_only:
        print(json.dumps(continue_experiment(config, args.source_experiment, output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
