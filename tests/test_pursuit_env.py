"""Environment API, baseline semantics, and complete random rollouts."""

from copy import deepcopy

import numpy as np
import pytest

from envs.pursuit_env import PursuitEnv


def test_reset_exposes_documented_observation_and_global_state_order(legacy_config_factory):
    initial = {
        "pursuer_positions": [[0.0, 0.0, 20.0], [10.0, 0.0, 20.0], [0.0, 20.0, 20.0]],
        "pursuer_velocities": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [-1.0, -2.0, -3.0]],
        "evader_position": [30.0, 30.0, 40.0],
        "evader_velocity": [3.0, 2.0, 1.0],
    }
    env = PursuitEnv(legacy_config_factory(initialization=initial))
    observation, info = env.reset(seed=5)
    expected = np.array([
        [1, 2, 3, 30, 30, 20, 2, 0, -2, 10, 0, 0, 0, 20, 0, 3, 3, 3, -2, -4, -6],
        [4, 5, 6, 20, 30, 20, -1, -3, -5, -10, 0, 0, -10, 20, 0, -3, -3, -3, -5, -7, -9],
        [-1, -2, -3, 30, 10, 20, 4, 4, 4, 0, -20, 0, 10, -20, 0, 2, 4, 6, 5, 7, 9],
    ])
    np.testing.assert_allclose(observation, expected)
    expected_state = np.array([[0, 0, 20, 1, 2, 3], [10, 0, 20, 4, 5, 6],
                               [0, 20, 20, -1, -2, -3], [30, 30, 40, 3, 2, 1]])
    np.testing.assert_allclose(env.get_state(), expected_state.ravel())
    np.testing.assert_allclose(info["global_state"], expected_state.ravel())
    np.testing.assert_allclose(env.states, expected_state)
    np.testing.assert_array_equal(env.accelerations, np.zeros((4, 3)))
    assert env.step_count == 0


def test_reward_is_shared_and_uses_configured_progress_and_time(legacy_config_factory):
    env = PursuitEnv(legacy_config_factory(
        evader={"max_horizontal_acceleration": 0.1},
        reward={"capture_reward": 71.0, "time_penalty": -0.2,
                "collision_penalty": -4.0, "progress_coefficient": 2.5},
    ))
    _, initial_info = env.reset(seed=0)
    observation, rewards, terminated, truncated, info = env.step(np.array([[1.0, 0.0, 0.0]] * 3))
    expected_progress = 2.5 * (initial_info["min_distance"] - info["min_distance"])
    assert expected_progress > 0.0
    np.testing.assert_allclose(rewards, [expected_progress - 0.2] * 3)
    assert info["reward_components"] == pytest.approx(
        {"capture": 0.0, "progress": expected_progress, "time": -0.2, "collision": 0.0}
    )
    assert observation.shape == (3, 21)
    assert rewards.shape == (3,)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert not terminated and not truncated
    assert info["step_count"] == env.step_count == 1
    assert not info["captured"]


def test_multiple_collisions_apply_one_team_penalty(legacy_config_factory):
    env = PursuitEnv(legacy_config_factory(
        initialization={"pursuer_positions": [[0, 0, 50], [0.5, 0, 50], [1, 0, 50]],
                        "evader_position": [60, 0, 50]},
        safety={"pursuer_safety_distance": 2.0},
        reward={"progress_coefficient": 0.0, "time_penalty": -0.2, "collision_penalty": -4.0},
    ))
    env.reset()
    _, rewards, terminated, truncated, info = env.step(np.zeros((3, 3)))
    assert info["collision_pairs"] == 3
    assert info["reward_components"]["collision"] == -4.0
    np.testing.assert_allclose(rewards, [-4.2] * 3)
    assert not terminated and not truncated


@pytest.mark.parametrize("distance,captured", [(4.99, True), (5.0, False), (5.01, False)])
def test_capture_radius_is_strict_and_capture_wins_over_timeout(legacy_config_factory, distance, captured):
    env = PursuitEnv(legacy_config_factory(
        environment={"max_episode_steps": 1},
        initialization={"pursuer_positions": [[0, 0, 50], [-30, -10, 50], [-30, 10, 50]],
                        "evader_position": [distance, 0, 50]},
        evader={"max_horizontal_acceleration": 1e-30, "max_vertical_acceleration": 1e-30},
        capture={"radius": 5.0},
        reward={"capture_reward": 71.0, "progress_coefficient": 0.0, "time_penalty": -0.2},
    ))
    env.reset()
    _, rewards, terminated, truncated, info = env.step(np.zeros((3, 3)))
    assert terminated is captured
    assert truncated is (not captured)
    assert info["captured"] is captured
    assert info["min_distance"] == pytest.approx(distance)
    assert info["reward_components"]["capture"] == (71.0 if captured else 0.0)
    np.testing.assert_allclose(rewards, [(71.0 if captured else 0.0) - 0.2] * 3)
    with pytest.raises(RuntimeError):
        env.step(np.zeros((3, 3)))


def test_configured_safety_distance_changes_collision_detection(legacy_config_factory):
    env = PursuitEnv(legacy_config_factory(
        initialization={"pursuer_positions": [[0, 0, 50], [1, 0, 50], [10, 0, 50]],
                        "evader_position": [60, 0, 50]},
        safety={"pursuer_safety_distance": 0.5},
    ))
    env.reset()
    _, _, _, _, info = env.step(np.zeros((3, 3)))
    assert info["collision_pairs"] == 0
    assert info["reward_components"]["collision"] == 0.0


def test_bounds_clip_positions_while_preserving_velocity(legacy_config_factory, legacy_config):
    policy = deepcopy(legacy_config["evader"]["policy"])
    policy.update(boundary_margin_horizontal=0.25, boundary_margin_vertical=0.25)
    env = PursuitEnv(legacy_config_factory(
        environment={"dt": 0.5, "bounds": {"low": [-1, -1, 0], "high": [1, 1, 2]}},
        initialization={"pursuer_positions": [[0.9, 0, 1], [-0.8, 0, 1], [-0.8, 0.8, 1]],
                        "pursuer_velocities": [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                        "evader_position": [0.8, 0.8, 1]},
        capture={"radius": 0.01},
        safety={"pursuer_safety_distance": 0.01},
        evader={"policy": policy},
    ))
    env.reset()
    _, _, terminated, truncated, _ = env.step(np.zeros((3, 3)))
    assert env.states[0, 0] == pytest.approx(1.0)
    assert env.states[0, 3] == pytest.approx(1.0)
    assert np.all(env.states[:, :3] >= [-1, -1, 0])
    assert np.all(env.states[:, :3] <= [1, 1, 2])
    assert not terminated and not truncated


def test_boundary_clip_diagnostics_count_aircraft_steps_and_reset(legacy_config_factory, legacy_config):
    low = np.asarray(legacy_config["environment"]["bounds"]["low"], dtype=float)
    high = np.asarray(legacy_config["environment"]["bounds"]["high"], dtype=float)
    center = (low + high) / 2.0
    p0 = high - 0.01
    p1 = np.array([low[0] + 0.01, center[1], center[2]])
    env = PursuitEnv(legacy_config_factory(initialization={
        "pursuer_positions": [p0.tolist(), p1.tolist(), center.tolist()],
        "pursuer_velocities": [[1, 1, 1], [-1, 0, 0], [0, 0, 0]],
        "evader_position": (center + [0.0, 100.0, 0.0]).tolist(),
        "evader_velocity": [0, 0, 0],
    }))
    _, reset_info = env.reset(seed=0)
    assert reset_info["boundary_clip_count"] == 0
    np.testing.assert_array_equal(reset_info["boundary_clip_counts"], np.zeros(4, dtype=np.int64))
    np.testing.assert_array_equal(reset_info["boundary_clipped"], np.zeros(4, dtype=bool))
    assert reset_info["boundary_clip_counts"].dtype == np.int64
    assert reset_info["boundary_clipped"].dtype == np.bool_
    _, _, terminated, truncated, first = env.step(np.zeros((3, 3)))
    assert not terminated and not truncated
    # P0 crosses three faces, but is one aircraft clipping event this step.
    assert first["boundary_clip_count"] == 2
    np.testing.assert_array_equal(first["boundary_clip_counts"], [1, 1, 0, 0])
    np.testing.assert_array_equal(first["boundary_clipped"], [True, True, False, False])
    np.testing.assert_array_equal(env.accelerations[:3], np.zeros((3, 3)))
    np.testing.assert_array_equal(env.states[:3, 3:], [[1, 1, 1], [-1, 0, 0], [0, 0, 0]])
    # Diagnostic arrays are copies; caller edits cannot corrupt future counts.
    first["boundary_clip_counts"][:] = 1000
    first["boundary_clipped"][:] = False
    _, _, _, _, second = env.step(np.zeros((3, 3)))
    assert second["boundary_clip_count"] == 4
    np.testing.assert_array_equal(second["boundary_clip_counts"], [2, 2, 0, 0])
    np.testing.assert_array_equal(second["boundary_clipped"], [True, True, False, False])
    np.testing.assert_array_equal(reset_info["boundary_clip_counts"], np.zeros(4))
    _, again = env.reset(seed=0)
    assert again["boundary_clip_count"] == 0
    np.testing.assert_array_equal(again["boundary_clip_counts"], np.zeros(4))
    np.testing.assert_array_equal(again["boundary_clipped"], np.zeros(4, dtype=bool))


def test_exact_boundary_without_overshoot_is_not_counted_or_steered_for_pursuers(
    legacy_config_factory, legacy_config
):
    high = legacy_config["environment"]["bounds"]["high"]
    env = PursuitEnv(legacy_config_factory(initialization={
        "pursuer_positions": [[high[0], 0, 75], [-200, -100, 75], [-200, 100, 75]],
        "pursuer_velocities": [[0, 0, 0]] * 3,
        "evader_position": [0, 0, 75],
        "evader_velocity": [0, 0, 0],
    }))
    env.reset()
    _, _, _, _, info = env.step(np.zeros((3, 3)))
    assert info["boundary_clip_count"] == 0
    np.testing.assert_array_equal(info["boundary_clipped"], np.zeros(4, dtype=bool))
    np.testing.assert_array_equal(env.states[0], [high[0], 0, 75, 0, 0, 0])
    np.testing.assert_array_equal(env.accelerations[:3], np.zeros((3, 3)))


def test_environment_enables_boundary_force_from_yaml(legacy_config_factory, legacy_config):
    env = PursuitEnv(legacy_config_factory())
    assert env.evader_policy.boundary_enabled
    high = np.asarray(legacy_config["environment"]["bounds"]["high"])
    force = env.evader_policy.boundary_repulsion(high)
    strength = legacy_config["evader"]["policy"]["boundary_strength"]
    np.testing.assert_allclose(force, [-strength] * 3)


def test_legacy_environment_config_keeps_boundary_steering_disabled(
    legacy_config_factory, legacy_config
):
    policy = {key: value for key, value in legacy_config["evader"]["policy"].items()
              if not key.startswith("boundary_")}
    env = PursuitEnv(legacy_config_factory(evader={"policy": policy}))
    assert not env.evader_policy.boundary_enabled
    env.reset(seed=0)
    observation, reward, _, _, _ = env.step(env.sample_actions())
    assert np.isfinite(observation).all()
    assert np.isfinite(reward).all()


def test_partial_environment_boundary_config_is_rejected(legacy_config_factory, legacy_config):
    policy = deepcopy(legacy_config["evader"]["policy"])
    del policy["boundary_strength"]
    with pytest.raises(ValueError):
        PursuitEnv(legacy_config_factory(evader={"policy": policy}))


def test_seed_reproduces_random_actions_and_trajectories():
    first, second = PursuitEnv(), PursuitEnv()
    first.reset(seed=42)
    second.reset(seed=42)
    first_action = None
    for _ in range(8):
        actions = first.sample_actions()
        np.testing.assert_array_equal(actions, second.sample_actions())
        if first_action is None:
            first_action = actions.copy()
        first_result, second_result = first.step(actions), second.step(actions)
        np.testing.assert_array_equal(first_result[0], second_result[0])
        np.testing.assert_array_equal(first_result[1], second_result[1])
        np.testing.assert_array_equal(first.get_state(), second.get_state())
    first.reset(seed=42)
    np.testing.assert_array_equal(first.sample_actions(), first_action)
    first.reset(seed=43)
    assert not np.array_equal(first.sample_actions(), first_action)
    assert first.step_count == 0
    np.testing.assert_array_equal(first.accelerations, np.zeros((4, 3)))


def test_step_requires_reset():
    with pytest.raises(RuntimeError):
        PursuitEnv().step(np.zeros((3, 3)))


@pytest.mark.parametrize("actions", [np.zeros(3), np.zeros((4, 3)), np.zeros((3, 2)),
                                    np.full((3, 3), np.nan), np.full((3, 3), np.inf),
                                    np.full((3, 3), 1.0 + 1.0j)])
def test_invalid_actions_are_rejected_without_advancing(actions):
    env = PursuitEnv()
    env.reset(seed=1)
    before = env.get_state().copy()
    with pytest.raises(ValueError):
        env.step(actions)
    assert env.step_count == 0
    np.testing.assert_array_equal(env.get_state(), before)


def test_out_of_range_actions_are_clipped():
    clipped, excessive = PursuitEnv(), PursuitEnv()
    clipped.reset(seed=0)
    excessive.reset(seed=0)
    clipped.step(np.array([[1.0, -1.0, 1.0]] * 3))
    excessive.step(np.array([[10.0, -10.0, 10.0]] * 3))
    np.testing.assert_allclose(clipped.get_state(), excessive.get_state())
    np.testing.assert_allclose(clipped.accelerations, excessive.accelerations)


@pytest.mark.parametrize("seed", [0, 7, 19])
def test_complete_random_policy_rollout_is_finite_and_physically_bounded(seed):
    env = PursuitEnv()
    observation, info = env.reset(seed=seed)
    dt = env.config["environment"]["dt"]
    low = np.asarray(env.config["environment"]["bounds"]["low"])
    high = np.asarray(env.config["environment"]["bounds"]["high"])
    max_steps = env.config["environment"]["max_episode_steps"]
    assert observation.shape == (3, 21)
    assert info["global_state"].shape == (24,)
    for step in range(1, max_steps + 1):
        previous_acceleration = env.accelerations.copy()
        actions = env.sample_actions()
        assert actions.shape == (3, 3)
        assert np.all(np.abs(actions) <= 1.0)
        observation, rewards, terminated, truncated, info = env.step(actions)
        assert observation.shape == (3, 21)
        assert rewards.shape == (3,)
        assert np.isfinite(observation).all()
        assert np.isfinite(rewards).all()
        assert np.isfinite(env.get_state()).all()
        assert np.isfinite(env.accelerations).all()
        assert np.all(env.states[:, :3] >= low)
        assert np.all(env.states[:, :3] <= high)
        adversary = env.config["attacker"] if env.mission_enabled else env.config["evader"]
        for agent, limits in enumerate([env.config["pursuer"]] * 3 + [adversary]):
            assert np.linalg.norm(env.states[agent, 3:5]) <= limits["max_horizontal_speed"] + 1e-10
            assert -limits["max_descent_speed"] - 1e-10 <= env.states[agent, 5] <= limits["max_climb_speed"] + 1e-10
            assert np.linalg.norm(env.accelerations[agent, :2]) <= limits["max_horizontal_acceleration"] + 1e-10
            assert abs(env.accelerations[agent, 2]) <= limits["max_vertical_acceleration"] + 1e-10
            change = env.accelerations[agent] - previous_acceleration[agent]
            assert np.linalg.norm(change) <= limits["max_jerk"] * dt + 1e-10
        assert env.step_count == step
        if terminated or truncated:
            assert terminated == (info["captured"] or info["attacker_reached_base"])
            assert terminated or step == max_steps
            break
    else:
        pytest.fail("Episode did not terminate or truncate at its configured limit")
    env.reset(seed=seed)
    assert env.step_count == 0
