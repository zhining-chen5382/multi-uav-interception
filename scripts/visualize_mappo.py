"""Save a static 3D trajectory figure from a deterministic MAPPO episode."""

import argparse
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import yaml

from utils.evaluation import (
    checkpoint_provenance, load_evaluation_run, run_episode, save_trajectory, write_json,
)
from utils.experiment import set_seed, write_metadata


def plot_trajectory(trajectory, metrics, path, figure_size, dpi):
    """Plot all four UAVs with equal physical scales and distinct endpoints."""
    positions = trajectory["positions"]
    mission_enabled = "base_position" in trajectory and "attack_radius" in trajectory
    figure = plt.figure(figsize=figure_size)
    axis = figure.add_subplot(111, projection="3d")
    colors = ("#0072B2", "#009E73", "#CC79A7", "#D55E00")
    labels = ("Pursuer 0", "Pursuer 1", "Pursuer 2", "Attacker" if mission_enabled else "Evader (historical)")
    styles = ("-", "--", "-.", "-")
    for index, (color, label, style) in enumerate(zip(colors, labels, styles)):
        path_positions = positions[:, index]
        axis.plot(
            path_positions[:, 0], path_positions[:, 1], path_positions[:, 2],
            color=color, linestyle=style, linewidth=2.0 if index < 3 else 2.5,
            label=label, alpha=0.9,
        )
        axis.scatter(
            *path_positions[0], marker="o", s=100, facecolors="white",
            edgecolors=color, linewidths=1.6, depthshade=False,
        )
        axis.scatter(
            *path_positions[-1], marker="X", s=55, color=color,
            edgecolors="white", linewidths=0.5, depthshade=False,
        )
    flattened = positions.reshape(-1, 3)
    if mission_enabled:
        base = np.asarray(trajectory["base_position"])
        radius = float(trajectory["attack_radius"])
        azimuth = np.linspace(0.0, 2.0 * np.pi, 33)
        polar = np.linspace(0.0, np.pi, 17)
        sphere = np.array([
            base[0] + radius * np.outer(np.cos(azimuth), np.sin(polar)),
            base[1] + radius * np.outer(np.sin(azimuth), np.sin(polar)),
            base[2] + radius * np.outer(np.ones_like(azimuth), np.cos(polar)),
        ])
        axis.plot_wireframe(*sphere, color="#B45A00", linewidth=0.45, alpha=0.35)
        axis.scatter(*base, marker="*", s=190, color="#222222", depthshade=False,
                     label="Base / defended point")
        flattened = np.vstack((flattened, base - radius, base + radius))
    low, high = flattened.min(axis=0), flattened.max(axis=0)
    span = high - low
    # A small minimum span keeps planar/collinear trajectories readable.
    minimum_span = max(float(span.max()) * 0.4, 1.0)
    shown_span = np.maximum(span * 1.12, minimum_span)
    center = (high + low) / 2.0
    for setter, index in zip((axis.set_xlim, axis.set_ylim, axis.set_zlim), range(3)):
        setter(center[index] - shown_span[index] / 2, center[index] + shown_span[index] / 2)
    if hasattr(axis, "set_box_aspect"):
        axis.set_box_aspect(shown_span)
    axis.set_xlabel("x (m)", labelpad=8)
    axis.set_ylabel("y (m)", labelpad=8)
    axis.set_zlabel("z (m)", labelpad=8)
    for coordinate_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
        coordinate_axis.set_major_locator(MaxNLocator(nbins=4))
    axis.view_init(elev=25, azim=-65)
    if mission_enabled:
        title = (
            "3A1C Point Defense | {} | steps={}\n"
            "Minimum pursuer-attacker distance={:.2f} m | final attacker-base distance={:.2f} m"
        ).format(metrics["outcome"], metrics["episode_length"], metrics["min_distance"],
                 metrics["attacker_base_terminal_distance"])
    else:
        title = "Historical 3A1C | {} | steps={}\nReward={:.2f} | minimum distance={:.2f} m".format(
            metrics.get("outcome", "Interception Success" if metrics["captured"] else "Timeout"),
            metrics["episode_length"], metrics["episode_reward"], metrics["min_distance"],
        )
    axis.set_title(title, pad=20, fontsize=11)
    handles, names = axis.get_legend_handles_labels()
    handles.extend([
        Line2D([], [], color="#555555", marker="o", markerfacecolor="white",
               linestyle="None", markersize=6),
        Line2D([], [], color="#555555", marker="X", linestyle="None", markersize=8),
    ])
    names.extend(["Start", "End"])
    if mission_enabled:
        handles.append(Line2D([], [], color="#B45A00", linewidth=1, alpha=0.65))
        names.append("Attack sphere (r={:g} m)".format(radius))
    axis.legend(handles, names, loc="upper left", framealpha=0.95)
    figure.subplots_adjust(left=0.04, right=0.92, bottom=0.06, top=0.86)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def visualize(checkpoint, output_dir=None, seed=None, env_config=None, device="cpu"):
    agent, payload, env, output_path, environment_source = load_evaluation_run(
        checkpoint, output_dir=output_dir, env_config=env_config, device=device,
        artifact_prefix="visualization",
    )
    config = deepcopy(payload["config"])
    visualization_config = config["visualization"]
    seed = visualization_config["seed"] if seed is None else seed
    set_seed(seed, config["torch_num_threads"])
    config["seed"] = seed
    config["device"] = device
    visualization_config["seed"] = seed
    with (output_path / "visualization_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
    metrics, trajectory = run_episode(env, agent, seed)
    save_trajectory(output_path / "trajectory.npz", trajectory, metrics, seed)
    write_json(output_path / "visualization_metrics.json", metrics)
    write_metadata(
        output_path / "visualization_metadata.json", config, env.config,
        config_path=(payload.get("metadata", {}).get("config_source")
                     or output_path / "visualization_config.yaml"),
        env_config_path=env.config_path, output_dir=output_path, kind="visualization",
        extra={
            "checkpoint": str(Path(checkpoint).resolve()),
            "checkpoint_training_state": checkpoint_provenance(payload),
            "environment_source": environment_source,
            "environment_override": env_config is not None,
            "deterministic": True,
        },
    )
    path = output_path / "trajectory_3d.png"
    plot_trajectory(
        trajectory, metrics, path,
        figure_size=visualization_config["figure_size"], dpi=visualization_config["dpi"],
    )
    print("Saved {} ({}, steps={})".format(
        path, metrics["outcome"], metrics["episode_length"]
    ))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--env-config", type=Path, help="Explicit environment configuration override")
    parser.add_argument("--device", default="cpu")
    visualize(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
