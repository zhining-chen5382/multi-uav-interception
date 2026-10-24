"""Validate Direct Attacker alone, stopping at the original attack sphere.

This deliberately uses one aircraft and the production policy/integrator,
without creating pursuers or an environment that could intercept the attacker.
Run from the repository root with ``python -m scripts.sanity_direct_attacker``.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import yaml

from envs.attacker_policy import DirectAttackerPolicy
from envs.dynamics import MotionLimits, integrate


ROOT = Path(__file__).resolve().parents[1]


def _vector(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("{} must be a finite three-vector".format(name))
    return result


def _positive(value, name):
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError("{} must be finite and positive".format(name))
    return result


def _check_limits(state, acceleration, previous_acceleration, dt, limits):
    """Audit applied commands and velocities, rather than clipping them again."""
    tolerance = 1e-9
    checks = {
        "finite state": np.isfinite(state).all(),
        "finite acceleration": np.isfinite(acceleration).all(),
        "horizontal speed": np.linalg.norm(state[3:5]) <= limits.max_horizontal_speed + tolerance,
        "climb speed": state[5] <= limits.max_climb_speed + tolerance,
        "descent speed": state[5] >= -limits.max_descent_speed - tolerance,
        "horizontal acceleration": (
            np.linalg.norm(acceleration[:2]) <= limits.max_horizontal_acceleration + tolerance
        ),
        "vertical acceleration": abs(acceleration[2]) <= limits.max_vertical_acceleration + tolerance,
        "jerk": np.linalg.norm(acceleration - previous_acceleration) <= limits.max_jerk * dt + tolerance,
    }
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise AssertionError("Sanity check physical/numerical failure: {}".format(", ".join(failed)))


def _sphere(axis, base, radius):
    azimuth = np.linspace(0.0, 2.0 * np.pi, 33)
    polar = np.linspace(0.0, np.pi, 17)
    coordinates = (
        base[0] + radius * np.outer(np.cos(azimuth), np.sin(polar)),
        base[1] + radius * np.outer(np.sin(azimuth), np.sin(polar)),
        base[2] + radius * np.outer(np.ones_like(azimuth), np.cos(polar)),
    )
    axis.plot_wireframe(*coordinates, color="#C17900", linewidth=0.5, alpha=0.4)
    axis.scatter(*base, marker="*", s=130, color="#222222", depthshade=False,
                 label="Base / defended point")


def _plot(trajectory, metrics, output):
    positions = trajectory["states"][:, :3]
    base = trajectory["base_position"]
    radius = float(trajectory["attack_radius"])
    figure = plt.figure(figsize=(13, 6))
    for index, zoom in enumerate((False, True), start=1):
        axis = figure.add_subplot(1, 2, index, projection="3d")
        path = positions[np.linalg.norm(positions - base, axis=1) <= radius * 4] if zoom else positions
        axis.plot(*path.T, color="#D55E00", linewidth=2.4, label="Direct Attacker")
        if not zoom:
            axis.scatter(*positions[0], marker="o", s=75, facecolors="white",
                         edgecolors="#D55E00", linewidths=1.8, depthshade=False, label="Start")
        axis.scatter(*positions[-1], marker="X", s=65, color="#D55E00", depthshade=False, label="End")
        _sphere(axis, base, radius)
        if zoom:
            low, high = base - radius * 3.5, base + radius * 3.5
            axis.set_title("Attack sphere detail (r={:g} m)".format(radius), pad=15)
        else:
            extent = np.vstack((positions, base - radius, base + radius))
            low, high = extent.min(axis=0), extent.max(axis=0)
            center = (low + high) / 2.0
            span = np.maximum((high - low) * 1.12, np.max(high - low) * 0.2)
            low, high = center - span / 2.0, center + span / 2.0
            axis.set_title("Full trajectory | no pursuers", pad=15)
            axis.legend(loc="upper left", fontsize=9, framealpha=0.95)
        for setter, coordinate in zip((axis.set_xlim, axis.set_ylim, axis.set_zlim), range(3)):
            setter(low[coordinate], high[coordinate])
        axis.set_box_aspect(high - low)
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_zlabel("z (m)")
        axis.view_init(elev=22, azim=-65)
        for coordinate_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
            coordinate_axis.set_major_locator(MaxNLocator(nbins=4))
    figure.suptitle(
        "Direct Attacker sanity check | {} | {} steps / {:.1f} s\n"
        "Base distance: {:.2f} -> {:.2f} m | entry speed: {} | hard clips: {}".format(
            metrics["outcome"], metrics["step_count"], metrics["duration_seconds"],
            metrics["initial_distance"], metrics["final_distance"],
            "{:.2f} m/s".format(metrics["entry_speed"]) if metrics["entry_speed"] is not None else "n/a",
            metrics["boundary_clip_count"],
        ), fontsize=12,
    )
    figure.subplots_adjust(left=0.02, right=0.94, bottom=0.08, top=0.81, wspace=0.1)
    figure.savefig(output / "trajectory_3d.png", dpi=160, facecolor="white")
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    time = trajectory["time"]
    velocity = trajectory["states"][:, 3:]
    axes[0].plot(time, trajectory["distance_to_base"], color="#D55E00", label="Raw base distance")
    axes[0].axhline(radius, color="#C17900", linestyle="--", label="Attack radius ({:g} m)".format(radius))
    axes[0].set_ylabel("Base distance (m)")
    axes[0].set_title("No pursuers | all recorded steps | no smoothing")
    axes[1].plot(time, np.linalg.norm(velocity, axis=1), label="Speed (3D)")
    axes[1].plot(time, np.linalg.norm(velocity[:, :2], axis=1), linestyle="--", label="Horizontal speed")
    axes[1].plot(time, velocity[:, 2], linestyle=":", label="Vertical velocity (signed)")
    axes[1].axhline(metrics["attack_speed"], color="#555555", linestyle="-.", label="Desired speed")
    axes[1].set_ylabel("Speed / velocity (m/s)")
    axes[1].set_xlabel("Simulation time (s)")
    for axis in axes:
        axis.legend(loc="best")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "distance_speed_raw.png", dpi=160, facecolor="white")
    plt.close(figure)


def _git_provenance():
    def read(*arguments):
        result = subprocess.run(
            ["git", "-c", "safe.directory={}".format(ROOT.as_posix())] + list(arguments),
            cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    try:
        commit, status = read("rev-parse", "HEAD"), read("status", "--porcelain")
    except (OSError, subprocess.TimeoutExpired):
        commit, status = None, None
    return {"code_version": commit, "git_dirty": None if status is None else bool(status)}


def sanity_check(config=ROOT / "configs" / "baseline_3a1c.yaml",
                 output_dir=ROOT / "outputs" / "direct_attacker_validation" / "sanity"):
    """Run exactly one deterministic single-aircraft episode, then save evidence."""
    config_path = Path(config).expanduser().resolve()
    config_bytes = config_path.read_bytes()
    settings = yaml.safe_load(config_bytes.decode("utf-8"))
    policy_settings = settings["attacker"]["policy"]
    if policy_settings.get("mode") != "direct":
        raise ValueError("Sanity check requires attacker.policy.mode: direct")
    base = _vector(settings["mission"]["base_position"], "mission.base_position")
    attack_radius = _positive(settings["mission"]["attack_radius"], "mission.attack_radius")
    environment = settings["environment"]
    dt = _positive(environment["dt"], "environment.dt")
    horizon = environment["max_episode_steps"]
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("environment.max_episode_steps must be a positive integer")
    bounds_low = _vector(environment["bounds"]["low"], "bounds.low")
    bounds_high = _vector(environment["bounds"]["high"], "bounds.high")
    if np.any(bounds_low >= bounds_high):
        raise ValueError("Each bounds.low must be less than bounds.high")
    limits = MotionLimits.from_config(settings["attacker"])
    policy = DirectAttackerPolicy(
        base_position=base, attack_speed=policy_settings["attack_speed"],
        velocity_gain=policy_settings["velocity_gain"], distance_epsilon=policy_settings["distance_epsilon"],
    )
    state = np.concatenate((
        _vector(settings["initialization"]["attacker_position"], "attacker_position"),
        _vector(settings["initialization"]["attacker_velocity"], "attacker_velocity"),
    ))
    if np.any(state[:3] < bounds_low) or np.any(state[:3] > bounds_high):
        raise ValueError("Initial attacker position must be inside numerical workspace")
    acceleration = np.zeros(3, dtype=np.float64)
    _check_limits(state, acceleration, acceleration, dt, limits)
    states, applied = [state.copy()], [acceleration.copy()]
    desired, commands, clips = [], [], []
    initial_distance = float(np.linalg.norm(state[:3] - base))
    if initial_distance < attack_radius:
        raise ValueError("Sanity check must start outside the attack sphere")
    for _ in range(horizon):
        desired_velocity = policy.desired_velocity(state)
        command = policy(state)
        if not np.isfinite(desired_velocity).all() or not np.isfinite(command).all():
            raise AssertionError("Non-finite Direct Attacker guidance")
        next_state, next_acceleration = integrate(state, command, acceleration, dt, limits)
        _check_limits(next_state, next_acceleration, acceleration, dt, limits)
        clipped = np.any((next_state[:3] < bounds_low) | (next_state[:3] > bounds_high))
        # Match the environment's existing numerical protection exactly.
        next_state[:3] = np.clip(next_state[:3], bounds_low, bounds_high)
        desired.append(desired_velocity.copy())
        commands.append(command.copy())
        clips.append(bool(clipped))
        states.append(next_state.copy())
        applied.append(next_acceleration.copy())
        state, acceleration = next_state, next_acceleration
        # Strict post-update sphere test, with no simulation after success.
        if np.linalg.norm(state[:3] - base) < attack_radius:
            break

    states = np.asarray(states)
    distances = np.linalg.norm(states[:, :3] - base, axis=1)
    steps = len(commands)
    reached_base = bool(distances[-1] < attack_radius)
    increase_steps = np.flatnonzero(np.diff(distances) > 1e-9) + 1
    initial_direction = (states[0, :3] - base) / initial_distance
    relative_positions = states[:, :3] - base
    cross_track = relative_positions - np.outer(relative_positions @ initial_direction, initial_direction)
    entry_velocity = states[-1, 3:] if reached_base else None
    metrics = {
        "policy_mode": "direct", "pursuers_present": False, "random_initialization": False,
        "attack_speed": float(policy_settings["attack_speed"]),
        "velocity_gain": float(policy_settings["velocity_gain"]),
        "attack_radius": attack_radius, "initial_distance": float(distances[0]),
        "minimum_distance": float(distances.min()), "final_distance": float(distances[-1]),
        "time_enter_attack_sphere": steps * dt if reached_base else None,
        "previous_distance_before_entry": float(distances[-2]) if reached_base else None,
        "strict_entry_verified": bool(reached_base and distances[-2] >= attack_radius),
        "entry_speed": float(np.linalg.norm(entry_velocity)) if reached_base else None,
        "entry_horizontal_speed": float(np.linalg.norm(entry_velocity[:2])) if reached_base else None,
        "entry_vertical_velocity": float(entry_velocity[2]) if reached_base else None,
        "boundary_clip_count": int(np.count_nonzero(clips)),
        "base_distance_increase_steps": increase_steps.tolist(),
        "base_distance_increase_count": int(len(increase_steps)),
        "maximum_cross_track_distance": float(np.linalg.norm(cross_track, axis=1).max()),
        "trajectory_length": float(np.linalg.norm(np.diff(states[:, :3], axis=0), axis=1).sum()),
        "step_count": steps, "max_episode_steps": horizon, "dt": dt,
        "duration_seconds": steps * dt, "enemy_attack_success": reached_base,
        "timeout": not reached_base, "outcome": "Enemy Attack Success" if reached_base else "Timeout",
        "all_values_finite": True, "physical_constraints_verified": True,
        "sanity_passed": bool(reached_base and not len(increase_steps) and not any(clips)),
    }
    trajectory = {
        "states": states, "applied_accelerations": np.asarray(applied),
        "desired_velocities": np.asarray(desired), "acceleration_commands": np.asarray(commands),
        "boundary_clipped": np.asarray(clips, dtype=bool), "time": np.arange(steps + 1) * dt,
        "distance_to_base": distances, "base_position": base, "attack_radius": np.asarray(attack_radius),
        "bounds_low": bounds_low, "bounds_high": bounds_high, "dt": np.asarray(dt),
    }
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to overwrite existing sanity artifacts: {}".format(output))
    output.mkdir(parents=True, exist_ok=True)
    (output / "environment_config.yaml").write_bytes(config_bytes)
    np.savez_compressed(output / "trajectory.npz", **trajectory)
    with (output / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    _plot(trajectory, metrics, output)
    manifest = {
        "kind": "direct_attacker_only_sanity_check", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_source": str(config_path), "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "environment_config": settings, "output_dir": str(output),
        "seed": None, "deterministic": True, "pursuers_present": False,
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("scripts/sanity_direct_attacker.py", "envs/attacker_policy.py", "envs/dynamics.py")
        },
        "npz_alignment": {
            "states_applied_accelerations_time_distance": "Initial state and every completed integration step (N+1)",
            "desired_velocities_acceleration_commands": "Evaluated at states[:-1], before each of N integrations",
            "boundary_clipped": "One boolean for each of N post-integration position proposals",
        },
        "plots": {"trajectory_3d.png": "Full trajectory and attack-sphere detail",
                  "distance_speed_raw.png": "Raw distance/speed samples; no smoothing"},
    }
    manifest.update(_git_provenance())
    with (output / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(metrics, ensure_ascii=False, allow_nan=False))
    print("Saved sanity evidence to {}".format(output))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "baseline_3a1c.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "direct_attacker_validation" / "sanity")
    metrics = sanity_check(**vars(parser.parse_args()))
    if not metrics["sanity_passed"]:
        raise SystemExit("Direct Attacker sanity check failed; see saved metrics and raw trajectory")


if __name__ == "__main__":
    main()
