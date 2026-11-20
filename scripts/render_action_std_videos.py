"""Render two already recorded action-std episodes, without policy or dynamics.

The deterministic and original-std examples are selected by the evaluation
protocol's lower-median minimum distance over all episodes, never by success.
This renderer reads arrays only and does not create or relabel checkpoints.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/v12_ActionStd敏感性/std_sensitivity_20261110_seed1000"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reference(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _load_episode(output, group, summary, representatives):
    selected = representatives[group]["typical"]
    trajectory_path = (output / selected["trajectory"]).resolve()
    trajectory_path.relative_to(output)
    checkpoint = Path(summary["checkpoint"]).resolve()
    config_path = output / "environment_config.yaml"
    protocol_path = output / "evaluation_protocol.yaml"
    protected = {
        str(path): _sha(path) for path in (
            checkpoint, trajectory_path, config_path, protocol_path,
            output / "summary.json", output / "representatives.json",
        )
    }
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    _require(protected[str(checkpoint)] == summary["checkpoint_sha256"], "Source checkpoint SHA mismatch")
    _require(summary["checkpoint_sha256"] == protocol["expected_checkpoint_sha256"], "Checkpoint differs from confirmed protocol")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    with np.load(trajectory_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files if name != "metadata_json"}
        metadata = json.loads(archive["metadata_json"].item())
    for name, array in arrays.items():
        if np.issubdtype(array.dtype, np.number):
            _require(np.isfinite(array).all(), "Non-finite recorded array: " + name)
    metrics = metadata["metrics"]
    _require(metrics == selected["metrics"], "Representative metrics differ from source NPZ")
    _require(metadata["seed"] == selected["seed"], "Representative seed differs from source NPZ")
    frames, dt = len(arrays["time"]), float(config["environment"]["dt"])
    _require(np.isfinite(dt) and dt > 0 and frames >= 2, "Invalid episode time sequence")
    states, times = arrays["states"], arrays["time"]
    _require(states.shape == (frames, 4, 6), "Expected all four aircraft and complete state history")
    _require(np.allclose(times, np.arange(frames) * dt, rtol=0, atol=1e-10), "Invalid physical timestamps")
    _require(metrics["episode_length"] == frames - 1, "Source episode length mismatch")
    _require(arrays["actions"].shape == (frames - 1, 3, 3), "Missing action transitions")
    _require(arrays["rewards"].shape == (frames - 1,), "Missing reward transitions")
    positions = states[:, :, :3]
    _require(np.array_equal(positions, arrays["positions"]), "State and position records differ")
    initialization = config["initialization"]
    expected = np.concatenate((np.vstack((initialization["pursuer_positions"], initialization["attacker_position"])),
                               np.vstack((initialization["pursuer_velocities"], initialization["attacker_velocity"]))), axis=1)
    _require(np.array_equal(states[0], expected), "Fixed initial state differs from saved environment")
    base = np.asarray(config["mission"]["base_position"], dtype=float)
    radius, capture_radius = float(config["mission"]["attack_radius"]), float(config["capture"]["radius"])
    distances = np.linalg.norm(positions[:, :3] - positions[:, 3:4], axis=2)
    nearest, base_distance = distances.min(axis=1), np.linalg.norm(positions[:, 3] - base, axis=1)
    captured = nearest < capture_radius
    attacked = (base_distance < radius) & ~captured
    _require(not np.any(captured[1:-1] | attacked[1:-1]), "Source continued beyond a task terminal state")
    timeout = not (captured[-1] or attacked[-1])
    _require(bool(metrics["captured"]) == bool(captured[-1])
             and bool(metrics["enemy_attack_success"]) == bool(attacked[-1])
             and bool(metrics["truncated"]) == timeout, "Saved task outcome mismatch")
    _require(not timeout or frames - 1 == config["environment"]["max_episode_steps"], "Invalid timeout duration")
    _require(np.isclose(nearest.min(), metrics["min_distance"])
             and np.isclose(base_distance[-1], metrics["attacker_base_terminal_distance"]), "Saved distance metric mismatch")
    _require(np.isclose(arrays["rewards"].sum(), metrics["episode_reward"]), "Saved episode return mismatch")
    lo = np.asarray(config["environment"]["bounds"]["low"])
    hi = np.asarray(config["environment"]["bounds"]["high"])
    proposals = positions[:-1] + states[1:, :, 3:] * dt
    _require(np.allclose(np.clip(proposals, lo, hi), positions[1:], rtol=1e-10, atol=1e-9),
             "Recorded positions violate the saved semi-implicit integration")
    group_summary = summary["groups"][group]
    multiplier = float(group_summary["multiplier"])
    _require(group_summary["deterministic"] is (multiplier == 0), "Sampling-mode metadata mismatch")
    _require(0 <= group_summary["capture_rate"] <= 1, "Invalid group capture rate")
    _require(summary["checkpoint_training_steps"] == protocol["expected_env_steps"] and summary["angle_degrees"] == 0,
             "Expected the confirmed source checkpoint and fixed zero-degree scenario")
    _require(summary["frozen_state_verification"]["all_checks_passed"]
             and summary["frozen_state_verification"]["no_training"], "Frozen evaluation verification did not pass")
    _require("lower median" in selected["selection"], "Expected the protocol's unfiltered typical selection")
    return dict(source=trajectory_path, checkpoint=checkpoint, config_path=config_path,
                protected=protected, metrics=metrics, selected=selected, config=config,
                group_summary=group_summary, multiplier=multiplier, states=states,
                positions=positions, times=times, dt=dt, base=base,
                radius=radius, capture_radius=capture_radius, nearest=nearest,
                base_distance=base_distance, reward_sum=np.r_[0, arrays["rewards"].cumsum()])


def _verify_decode(video, manifest, qa):
    reader = imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24", output_params=["-threads", "1"])
    samples, count = {}, 0
    try:
        decoded = next(reader)
        width, height = decoded["size"]
        _require([width, height] == manifest["resolution"] and np.isclose(decoded["fps"], manifest["fps"]),
                 "Decoded resolution/FPS mismatch")
        _require(abs(decoded["duration"] - manifest["video_seconds"]) < 0.02, "Decoded duration mismatch")
        indices = {0, manifest["frames"] // 2, manifest["frames"] - 1}
        for index, frame in enumerate(reader):
            _require(len(frame) == width * height * 3, "Decoded frame is incomplete")
            if index in indices:
                rgb = np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 3)
                source = np.asarray(Image.open(qa / "source_frame_{:03d}.png".format(index)).convert("RGB"))
                error = float(np.abs(rgb.astype(float) - source.astype(float)).mean())
                _require(error < 5.0 and rgb.std() > 10, "Decoded frame is blank or differs from source")
                samples[str(index)] = dict(mean_absolute_RGB_error_vs_source=error, nonblank=True)
                Image.fromarray(rgb).save(qa / "decoded_frame_{:03d}.png".format(index))
            count += 1
        _require(count == manifest["frames"] and len(samples) == len(indices), "Initial/middle/final frames missing")
    finally:
        reader.close()
    _require(_sha(video) == manifest["video_sha256"], "Encoded video SHA changed")
    for filename, expected in manifest["protected_sources_sha256"].items():
        _require(_sha(filename) == expected, "Source artifact changed: " + filename)
    return dict(created_at_utc=datetime.now(timezone.utc).isoformat(), all_checks_passed=True,
                decoded_frames=count, samples=samples, source_artifacts_unchanged=True,
                includes_initial_and_final_states=True, interpolation=False,
                simulation_rerun=False, renderer_runs_policy_or_dynamics=False)


def _render_group(output, group, summary, representatives):
    data = _load_episode(output, group, summary, representatives)
    folder = output / "videos" / group
    if folder.exists():
        raise FileExistsError("Refusing to overwrite existing video artifacts: " + str(folder))
    qa = folder / "qa"
    qa.mkdir(parents=True)
    target, temporary = folder / "episode.mp4", folder / "episode.partial.mp4"
    log_path = folder / "encoder.log"
    positions, times, metrics = data["positions"], data["times"], data["metrics"]
    frames, fps = len(times), 1.0 / data["dt"]
    width, height, dpi = 1280, 720, 100
    colors = ("#0072B2", "#009E73", "#CC79A7", "#D55E00")
    styles, labels = ("-", "--", "-.", "-"), ("Pursuer 0", "Pursuer 1", "Pursuer 2", "Direct Attacker")
    figure = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="white")
    main = figure.add_axes([0.025, 0.13, 0.63, 0.68], projection="3d")
    xy = figure.add_axes([0.73, 0.55, 0.245, 0.25])
    base, radius = data["base"], data["radius"]
    extent = np.vstack((positions.reshape(-1, 3), base - radius, base + radius))
    low, high = extent.min(axis=0), extent.max(axis=0)
    span = np.maximum((high - low) * 1.15, max(float((high - low).max()), 1.0) * 0.3)
    center = (high + low) / 2
    limits = np.column_stack((center - span / 2, center + span / 2))
    for setter, index in zip((main.set_xlim, main.set_ylim, main.set_zlim), range(3)):
        setter(*limits[index])
    main.set_box_aspect(span)
    main.view_init(elev=25, azim=-65)
    for axis, label in zip((main.xaxis, main.yaxis, main.zaxis), ("x (m)", "y (m)", "z (m)")):
        axis.set_major_locator(MaxNLocator(nbins=4))
        axis.set_label_text(label)
        axis.labelpad = 8
    azimuth, polar = np.linspace(0, 2 * np.pi, 25), np.linspace(0, np.pi, 13)
    sphere = (base[0] + radius * np.outer(np.cos(azimuth), np.sin(polar)),
              base[1] + radius * np.outer(np.sin(azimuth), np.sin(polar)),
              base[2] + radius * np.outer(np.ones_like(azimuth), np.cos(polar)))
    main.plot_wireframe(*sphere, color="#B87400", alpha=0.35, linewidth=0.5)
    main.plot(*[[value] for value in base], marker="*", color="#222222", markersize=10, linestyle="None")
    xy.set_xlim(*limits[0])
    xy.set_ylim(*limits[1])
    xy.set_aspect("equal", adjustable="box")
    xy.grid(alpha=0.2)
    xy.set_title("XY top view", fontsize=10)
    xy.set_xlabel("x (m)", fontsize=9)
    xy.set_ylabel("y (m)", fontsize=9)
    xy.tick_params(labelsize=8)
    xy.xaxis.set_major_locator(MaxNLocator(nbins=4))
    xy.yaxis.set_major_locator(MaxNLocator(nbins=3))
    xy.plot(base[0], base[1], marker="*", markersize=9, color="#222222", linestyle="None")
    xy.add_patch(plt.Circle(base[:2], radius, fill=False, color="#B87400", alpha=0.6))
    main_lines, main_markers, main_ends, xy_lines, xy_markers, xy_ends = [], [], [], [], [], []
    for index, (color, style) in enumerate(zip(colors, styles)):
        main_lines.append(main.plot([], [], [], color=color, linestyle=style, linewidth=2)[0])
        main_markers.append(main.plot([], [], [], marker="o", color=color, markersize=6, linestyle="None")[0])
        main_ends.append(main.plot([], [], [], marker="X", color=color, markersize=8, linestyle="None")[0])
        main.plot(*[[value] for value in positions[0, index]], marker="o", markerfacecolor="white",
                  markeredgecolor=color, markersize=7, linestyle="None")
        xy_lines.append(xy.plot([], [], color=color, linestyle=style, linewidth=1.4)[0])
        xy_markers.append(xy.plot([], [], marker="o", color=color, markersize=4, linestyle="None")[0])
        xy_ends.append(xy.plot([], [], marker="X", color=color, markersize=6, linestyle="None")[0])
        xy.plot(*positions[0, index, :2], marker="o", markerfacecolor="white", markeredgecolor=color,
                markersize=4, linestyle="None")
    handles = [Line2D([], [], color=color, linestyle=style, linewidth=2, label=label)
               for color, style, label in zip(colors, styles, labels)]
    handles.extend([Line2D([], [], color="#222222", marker="*", linestyle="None", label="Base"),
                    Line2D([], [], color="#555555", marker="o", markerfacecolor="white", linestyle="None", label="Start"),
                    Line2D([], [], color="#555555", marker="X", linestyle="None", label="End")])
    figure.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.03, 0.035), ncol=4, frameon=False, fontsize=9)
    multiplier = data["multiplier"]
    sampling = "Deterministic" if multiplier == 0 else "Stochastic"
    figure.text(0.04, 0.965, "Frozen v08 | std {:g}x | {}".format(multiplier, sampling),
                fontsize=18, weight="bold", va="top")
    figure.text(0.04, 0.916,
                "Checkpoint {:,} steps | Angle 0 deg | Seed {} | Group capture {:.0%} ({:d} episodes)".format(
                    summary["checkpoint_training_steps"], data["selected"]["seed"],
                    data["group_summary"]["capture_rate"], data["group_summary"]["episodes"]), fontsize=11.5)
    figure.text(0.04, 0.875, "Typical = lower-median minimum distance over ALL episodes; no success filtering",
                fontsize=10, color="#444444")
    clock = figure.text(0.04, 0.825, "", fontsize=11, color="#333333")
    stats = figure.text(0.72, 0.445, "", va="top", fontsize=10.5, family="monospace", linespacing=1.5)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
               "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-s", "{}x{}".format(width, height),
               "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
               "-pix_fmt", "yuv420p", "-threads", "2", "-movflags", "+faststart", str(temporary)]
    with log_path.open("xb") as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
        try:
            for frame in range(frames):
                for index in range(4):
                    path, point = positions[:frame + 1, index], positions[frame, index]
                    main_lines[index].set_data_3d(path[:, 0], path[:, 1], path[:, 2])
                    main_markers[index].set_data_3d(*[[value] for value in point])
                    xy_lines[index].set_data(path[:, 0], path[:, 1])
                    xy_markers[index].set_data([point[0]], [point[1]])
                    if frame == frames - 1:
                        main_ends[index].set_data_3d(*[[value] for value in point])
                        xy_ends[index].set_data([point[0]], [point[1]])
                finished = frame == frames - 1
                status = metrics["outcome"].upper() if finished else "RUNNING"
                clock.set_text("Time {:4.1f} / {:.1f} s | Step {} / {} | Playback 1.0x".format(
                    times[frame], times[-1], frame, frames - 1))
                stats.set_text("{}\n\nBase dist: {:7.2f} m\nNearest P: {:7.2f} m\nClosest:   {:7.2f} m\nCapture: {} (<{:g} m)\nReturn: {:9.2f}".format(
                    status, data["base_distance"][frame], data["nearest"][frame],
                    data["nearest"][:frame + 1].min(), "YES" if data["nearest"][frame] < data["capture_radius"] else "no",
                    data["capture_radius"], data["reward_sum"][frame]))
                stats.set_color(("#137333" if metrics["captured"] else "#B22222") if finished else "#222222")
                figure.canvas.draw()
                rgb = np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy()
                if frame in (0, frames // 2, frames - 1):
                    Image.fromarray(rgb).save(qa / "source_frame_{:03d}.png".format(frame))
                process.stdin.write(rgb.tobytes())
                if frame % 100 == 0:
                    print("{}: rendered {}/{} frames".format(group, frame, frames), flush=True)
            process.stdin.close()
            if process.wait():
                raise RuntimeError("ffmpeg failed; inspect " + str(log_path))
        except BaseException:
            if process.poll() is None:
                process.kill()
                process.wait()
            raise
        finally:
            plt.close(figure)
    manifest = dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(), group=group,
        multiplier=multiplier, deterministic=multiplier == 0,
        angle_degrees=0, checkpoint=_reference(data["checkpoint"]),
        checkpoint_sha256=summary["checkpoint_sha256"], checkpoint_training_steps=summary["checkpoint_training_steps"],
        source_trajectory=_reference(data["source"]), source_sha256=data["protected"][str(data["source"])],
        environment_config=_reference(data["config_path"]),
        config_sha256=data["protected"][str(data["config_path"])],
        selection=data["selected"]["selection"], episode_index=data["selected"]["episode_index"],
        seed=data["selected"]["seed"], episode_metrics=metrics,
        reference_group_capture_rate=data["group_summary"]["capture_rate"],
        reference_group_episodes=data["group_summary"]["episodes"],
        base_std=data["group_summary"]["action_std"]["base_std"],
        scaled_std=data["group_summary"]["action_std"]["scaled_std"],
        frames=frames, fps=fps, resolution=[width, height],
        simulation_seconds=float(times[-1]), video_seconds=frames / fps,
        playback_speed=fps * data["dt"], includes_initial_and_final_states=True,
        interpolation=False, simulation_rerun=False, renderer_runs_policy_or_dynamics=False,
        protected_sources_sha256=data["protected"], output=_reference(target),
        video_sha256=_sha(temporary), encoder_command=command,
    )
    verification = _verify_decode(temporary, manifest, qa)
    temporary.replace(target)
    _write_json(folder / "video_manifest.json", manifest)
    _write_json(folder / "video_verification.json", verification)
    print("Saved {} ({} fully decoded frames)".format(target, frames), flush=True)
    return manifest


def render(output_dir=DEFAULT_OUTPUT):
    output = Path(output_dir).resolve()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    representatives = json.loads((output / "representatives.json").read_text(encoding="utf-8"))
    delivery = output / "videos" / "video_delivery_verification.json"
    if delivery.exists():
        raise FileExistsError("Refusing to overwrite video delivery verification")
    groups = []
    for multiplier in (0, 1):
        matches = [name for name, value in summary["groups"].items() if value["multiplier"] == multiplier]
        _require(len(matches) == 1, "Require exactly one zero-std and one original-std group")
        groups.append(matches[0])
    for group in groups:
        if (output / "videos" / group).exists():
            raise FileExistsError("Refusing to overwrite existing video directory: " + group)
    manifests = [_render_group(output, group, summary, representatives) for group in groups]
    result = dict(all_checks_passed=True, same_frozen_source_checkpoint=True,
                  no_training=True, no_inference=True, no_dynamics=True,
                  representative_selection="lower median minimum distance over all episodes, without outcome filtering",
                  videos=[dict(group=item["group"], multiplier=item["multiplier"],
                               deterministic=item["deterministic"], seed=item["seed"],
                               path=item["output"], frames=item["frames"],
                               outcome=item["episode_metrics"]["outcome"]) for item in manifests])
    _write_json(delivery, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    render(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
