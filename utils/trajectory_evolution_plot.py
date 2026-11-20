"""Publication-style XY views of frozen policies over actual training steps."""

from math import ceil
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


STYLES = ("-", "--", ":", "-.")
LABELS = ("Pursuer 0", "Pursuer 1", "Pursuer 2", "Direct Attacker")


def _bounds(episodes):
    xy = np.concatenate([entry["trajectory"]["positions"][:, :, :2].reshape(-1, 2) for entry in episodes])
    bases = [entry["trajectory"]["base_position"][:2] for entry in episodes if "base_position" in entry["trajectory"]]
    if bases:
        xy = np.vstack((xy, bases))
    low, high = xy.min(axis=0), xy.max(axis=0)
    padding = max(float(np.max(high - low)) * 0.06, 1.0)
    return low - padding, high + padding


def _decorate(axis, trajectory, bounds):
    low, high = bounds
    axis.set_xlim(low[0], high[0])
    axis.set_ylim(low[1], high[1])
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.grid(alpha=0.18)
    axis.set_axisbelow(True)
    if "base_position" in trajectory:
        base = trajectory["base_position"]
        axis.scatter(base[0], base[1], marker="*", s=150, c="#111111", zorder=12)
        if "attack_radius" in trajectory:
            axis.add_patch(plt.Circle(base[:2], float(trajectory["attack_radius"]), fill=False,
                                      color="#555555", linewidth=0.8, alpha=0.65))


def _draw(axis, entry, color, agent_indices=range(4)):
    positions = entry["trajectory"]["positions"]
    for agent in agent_indices:
        xy = positions[:, agent, :2]
        axis.plot(xy[:, 0], xy[:, 1], color=color, linestyle=STYLES[agent],
                  linewidth=1.7 if agent < 3 else 1.0, alpha=0.9, zorder=3)
        axis.scatter(*xy[-1], color=color, marker="x", s=22, linewidths=1.0, zorder=5)
    if entry["metrics"]["captured"]:
        point = positions[-1, -1, :2]
        axis.scatter(*point, s=100, marker="o", facecolors="none", edgecolors="#087F23",
                     linewidths=1.7, zorder=10)


def _start_points(axis, trajectory):
    initial = trajectory["positions"][0]
    axis.scatter(initial[:, 0], initial[:, 1], marker="o", facecolors="white",
                 edgecolors="#444444", s=32, linewidths=1.1, zorder=11)


def plot_evolution(episodes, output_dir, config):
    """Keep the overlay as the main figure; add small multiples for readability."""
    output = Path(output_dir)
    bounds = _bounds(episodes)
    steps = np.asarray([entry["step"] for entry in episodes], dtype=float)
    fraction = (steps - steps.min()) / max(float(np.ptp(steps)), 1.0)
    shades = float(config["shade_min"]) + fraction * (float(config["shade_max"]) - float(config["shade_min"]))
    cmap = plt.get_cmap(config["colormap"])
    colors = [cmap(value) for value in shades]
    match = re.search(r"v\d+-r\d+", str(episodes[0]["checkpoint"]))
    run_label = match.group(0) if match else "MAPPO"
    subtitle = "{} | angle {:+g} deg | seed {} | light = early, dark = late".format(
        run_label, config["angle_degrees"], config["seed"])
    figure = plt.figure(figsize=config["figure_size"])
    axis = figure.add_axes([0.065, 0.25, 0.60, 0.66])
    _decorate(axis, episodes[0]["trajectory"], bounds)
    for entry, color in zip(episodes, colors):
        _draw(axis, entry, color)
    _start_points(axis, episodes[0]["trajectory"])
    agent_handles = [Line2D([], [], color="#333333", linestyle=style, linewidth=1.5, label=name)
                     for style, name in zip(STYLES, LABELS)]
    agent_handles += [
        Line2D([], [], color="#111111", marker="*", linestyle="None", markersize=9, label="Base"),
        Line2D([], [], color="#444444", marker="o", markerfacecolor="white", linestyle="None", label="Start"),
        Line2D([], [], color="#444444", marker="x", linestyle="None", label="End"),
        Line2D([], [], color="#087F23", marker="o", markerfacecolor="none", linestyle="None", label="Capture event (3D)"),
    ]
    axis.legend(handles=agent_handles, loc="best", ncol=2, fontsize=8, framealpha=0.95)
    table_axis = figure.add_axes([0.705, 0.29, 0.28, 0.58])
    table_axis.axis("off")
    table_rows = []
    for entry in episodes:
        metrics = entry["metrics"]
        table_rows.append(["{:,}".format(entry["step"]), "CAPTURE" if metrics["captured"] else "FAIL",
                           "{:.1f}".format(float(entry["trajectory"]["time"][-1])),
                           "{:.2f}".format(metrics["min_distance"])])
    table = table_axis.table(cellText=table_rows, colLabels=[config.get("step_label", "Actual step"), "Result", "Time (s)", "Min d (m)"],
                             cellLoc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for column in range(4):
        table[0, column].set_facecolor("#E9EDF2")
        table[0, column].set_text_props(weight="bold")
    for row, (entry, color) in enumerate(zip(episodes, colors), start=1):
        table[row, 0].set_facecolor(color)
        table[row, 0].get_text().set_color("white" if shades[row-1] > 0.65 else "#111111")
        table[row, 1].get_text().set_color("#087F23" if entry["metrics"]["captured"] else "#A32323")
    checkpoint_handles = [Line2D([], [], color=color, linewidth=2.5,
                                label="{:,} | {}".format(entry["step"], "CAPTURE" if entry["metrics"]["captured"] else "FAIL"))
                          for entry, color in zip(episodes, colors)]
    figure.legend(handles=checkpoint_handles, loc="lower left", bbox_to_anchor=(0.065, 0.055),
                   ncol=min(4, len(episodes)), frameon=False, fontsize=8.5)
    figure.suptitle("Trajectory evolution - XY top view", y=0.978, fontsize=18, weight="bold")
    figure.text(0.5, 0.936, subtitle, ha="center", fontsize=11)
    figure.text(0.065, 0.025, "Results and minimum distance use full 3D states; XY crossings alone do not imply capture.",
                fontsize=9, color="#444444")
    if config.get("subtitle_note"):
        figure.text(0.065, 0.008, config["subtitle_note"], fontsize=8, color="#444444")
    main = output / "trajectory_evolution_xy.png"
    figure.savefig(main, dpi=config["dpi"], facecolor="white")
    plt.close(figure)

    columns = min(4, len(episodes))
    rows = int(ceil(len(episodes) / columns))
    grid, axes = plt.subplots(rows, columns, figsize=(4.1 * columns, 4.0 * rows), squeeze=False)
    for index, (entry, color) in enumerate(zip(episodes, colors)):
        panel = axes.flat[index]
        _decorate(panel, entry["trajectory"], bounds)
        _draw(panel, entry, color)
        _start_points(panel, entry["trajectory"])
        metrics = entry["metrics"]
        panel.set_title("{:,} steps | {}\nt={:.1f}s, 3D min d={:.2f}m".format(
            entry["step"], "CAPTURE" if metrics["captured"] else "FAIL",
            entry["trajectory"]["time"][-1], metrics["min_distance"]), fontsize=9)
        panel.tick_params(labelsize=8)
    for panel in list(axes.flat)[len(episodes):]:
        panel.set_visible(False)
    grid.suptitle("Trajectory evolution by checkpoint | " + subtitle, fontsize=12)
    if config.get("subtitle_note"):
        grid.text(0.02, 0.006, config["subtitle_note"], fontsize=8, color="#444444")
    grid.legend(handles=agent_handles, loc="lower center", ncol=4, frameon=False, fontsize=8)
    grid.tight_layout(rect=[0, 0.055, 1, 0.95], h_pad=3.0, w_pad=1.5)
    multiples = output / "trajectory_evolution_checkpoints_xy.png"
    grid.savefig(multiples, dpi=config["dpi"], facecolor="white")
    plt.close(grid)
    return {"xy": str(main.resolve()), "checkpoints_xy": str(multiples.resolve())}
