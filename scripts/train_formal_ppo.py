"""Train fresh independent formal PPO runs, or resume one into a new directory."""
import argparse
import json
from pathlib import Path

from utils.formal_ppo_training import load_configuration, train, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ppo_formal_v1.yaml"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-at-steps", type=int)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    config = load_configuration(args.config)
    validate_config(config)
    output = args.output_dir or Path(config.get("output_dir", config["experiment"]["output_root"]))
    if args.check_config:
        print(json.dumps(dict(configuration_valid=True, seeds=config["experiment"]["seeds"],
                              total_env_steps=config["training"]["total_env_steps"],
                              num_envs=config["training"]["num_envs"], output_dir=str(output)), indent=2))
        return
    if args.resume is not None and args.seed is None:
        parser.error("--resume requires the checkpoint's original --seed")
    from utils.formal_ppo_reporting import postprocess_run
    if args.seed is not None:
        summaries = [train(config, output, args.seed, resume_checkpoint=args.resume,
                           stop_at_steps=args.stop_at_steps, post_training_fn=postprocess_run)]
    else:
        if output.exists():
            parser.error("Use a new experiment output directory")
        output.mkdir(parents=True)
        summaries = []
        for seed in config["experiment"]["seeds"]:
            summaries.append(train(config, output / ("seed_{}".format(seed)), seed,
                                   stop_at_steps=args.stop_at_steps, post_training_fn=postprocess_run))
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
