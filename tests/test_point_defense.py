"""Point-defense initialization, strict post-step outcomes and task invariants."""

import numpy as np
import pytest

from envs import PursuitEnv


def _quiet_aircraft():
    # Negligible acceleration keeps representable threshold positions unchanged;
    # dynamics still runs, rather than replacing step or outcome calculations.
    return {"max_horizontal_acceleration": 1e-30,
            "max_vertical_acceleration": 1e-30}


def _terminal_case(config_factory, attacker_position, pursuer_position=None, steps=1):
    if pursuer_position is None:
        pursuer_position = [-40.0, 0.0, 50.0]
    return PursuitEnv(config_factory(
        environment={"max_episode_steps": steps},
        initialization={
            "pursuer_positions": [pursuer_position, [-40.0, -20.0, 50.0],
                                  [-40.0, 20.0, 50.0]],
            "pursuer_velocities": [[0.0, 0.0, 0.0]] * 3,
            "attacker_position": attacker_position,
            "attacker_velocity": [0.0, 0.0, 0.0],
        },
        attacker=_quiet_aircraft(),
    ))


def test_default_point_defense_initialization_has_no_collision_and_preserves_inputs(
    baseline_config,
):
    env = PursuitEnv()
    observation, info = env.reset(seed=7)
    base = np.asarray(baseline_config["mission"]["base_position"])
    assert env.mission_enabled
    np.testing.assert_array_equal(base, [0.0, 0.0, 0.0])
    assert baseline_config["mission"]["detection_radius"] == 500.0
    assert baseline_config["mission"]["attack_radius"] == 10.0
    assert env.capture_radius == 5.0
    assert np.all(np.linalg.norm(env.states[:3, :3] - base, axis=1) <= 2.0)
    pairwise = [np.linalg.norm(env.states[i, :3] - env.states[j, :3])
                for i in range(3) for j in range(i + 1, 3)]
    assert min(pairwise) >= env.safety_distance
    assert info["collision_pairs"] == 0
    assert np.linalg.norm(env.states[-1, :3] - base) == pytest.approx(500.0)
    assert info["distance_to_base"] == pytest.approx(500.0)
    assert not info["interception_success"]
    assert not info["attacker_reached_base"]
    assert observation.shape == (3, 21)
    assert info["global_state"].shape == env.get_state().shape == (24,)
    np.testing.assert_array_equal(env.states[:, 3:], np.zeros((4, 3)))
    assert np.isfinite(observation).all()
    assert np.isfinite(env.get_state()).all()
    assert not info["boundary_steering_enabled"]
    assert not env.attacker_policy.boundary_enabled
    assert info["boundary_clip_count"] == 0


def test_point_defense_keeps_original_physical_limits_and_horizon(baseline_config):
    assert baseline_config["environment"]["dt"] == 0.1
    assert baseline_config["environment"]["max_episode_steps"] == 500
    expected = {
        "max_horizontal_speed": 20.0, "max_climb_speed": 8.0,
        "max_descent_speed": 6.0, "max_horizontal_acceleration": 3.0,
        "max_vertical_acceleration": 2.0, "max_jerk": 5.0,
    }
    for aircraft in ("pursuer", "attacker"):
        assert {key: baseline_config[aircraft][key] for key in expected} == expected
    low = np.asarray(baseline_config["environment"]["bounds"]["low"])
    high = np.asarray(baseline_config["environment"]["bounds"]["high"])
    assert np.all(low < -600.0) and np.all(high > 600.0)


@pytest.mark.parametrize("distance,failed", [(9.99, True), (10.0, False), (10.01, False)])
def test_attack_sphere_is_strict_and_failure_has_configured_team_reward(
    config_factory, baseline_config, distance, failed,
):
    env = _terminal_case(config_factory, [distance, 0.0, 0.0])
    _, initial = env.reset(seed=0)
    _, rewards, terminated, truncated, info = env.step(np.zeros((3, 3)))
    assert terminated is failed
    assert truncated is (not failed)
    assert info["attacker_reached_base"] is failed
    assert info["enemy_attack_success"] is failed
    assert info["interception_failure"] is failed
    assert not info["captured"] and not info["interception_success"]
    assert info["distance_to_base"] == pytest.approx(distance)
    coefficients = baseline_config["reward"]
    expected_failure = coefficients["attack_failure_penalty"] if failed else 0.0
    expected_progress = coefficients["progress_coefficient"] * (
        initial["min_distance"] - info["min_distance"]
    )
    components = info["reward_components"]
    assert components["attack_failure"] == expected_failure
    assert components["capture"] == 0.0
    np.testing.assert_allclose(
        rewards, [expected_failure + expected_progress + coefficients["time_penalty"]] * 3
    )
    with pytest.raises(RuntimeError):
        env.step(np.zeros((3, 3)))


@pytest.mark.parametrize("distance,captured", [(4.99, True), (5.0, False), (5.01, False)])
def test_point_defense_capture_still_uses_strict_five_metre_threshold(
    config_factory, distance, captured,
):
    env = _terminal_case(
        config_factory, [50.0, 0.0, 50.0], [50.0 - distance, 0.0, 50.0]
    )
    env.reset(seed=0)
    _, rewards, terminated, truncated, info = env.step(np.zeros((3, 3)))
    assert terminated is captured
    assert truncated is (not captured)
    assert info["captured"] is captured
    assert info["interception_success"] is captured
    assert not info["attacker_reached_base"]
    assert not info["enemy_attack_success"]
    assert not info["interception_failure"]
    assert info["reward_components"]["capture"] == (100.0 if captured else 0.0)
    assert info["reward_components"]["attack_failure"] == 0.0
    assert np.isfinite(rewards).all()


@pytest.mark.parametrize("max_steps", [1, 500])
def test_simultaneous_capture_and_attack_uses_capture_priority(config_factory, max_steps):
    env = _terminal_case(config_factory, [9.0, 0.0, 0.0], [5.0, 0.0, 0.0], max_steps)
    env.reset(seed=0)
    _, rewards, terminated, truncated, info = env.step(np.zeros((3, 3)))
    assert info["distance_to_base"] < 10.0
    assert info["min_distance"] < 5.0
    assert terminated and not truncated
    assert info["captured"] and info["interception_success"]
    assert not info["attacker_reached_base"]
    assert not info["enemy_attack_success"]
    assert not info["interception_failure"]
    assert info["reward_components"]["capture"] == 100.0
    assert info["reward_components"]["attack_failure"] == 0.0
    np.testing.assert_allclose(rewards, [99.99] * 3)


def test_attack_failure_penalty_is_read_from_yaml(config_factory):
    path = config_factory(
        initialization={"attacker_position": [9.0, 0.0, 0.0],
                        "pursuer_positions": [[-40, 0, 50], [-40, -20, 50], [-40, 20, 50]]},
        attacker=_quiet_aircraft(),
        reward={"attack_failure_penalty": -73.0, "progress_coefficient": 0.0},
    )
    env = PursuitEnv(path)
    env.reset()
    _, reward, terminated, truncated, info = env.step(np.zeros((3, 3)))
    assert terminated and not truncated
    assert info["reward_components"]["attack_failure"] == -73.0
    np.testing.assert_allclose(reward, [-73.01] * 3)


@pytest.mark.parametrize("position,failed", [
    ([6.0, 8.0, 0.0], False),
    ([0.0, 0.0, 10.0], False),
    ([0.0, 0.0, 9.0], True),
    ([9.0, 0.0, 9.0], False),
])
def test_attack_region_uses_three_dimensional_spherical_distance(
    config_factory, position, failed,
):
    env = _terminal_case(config_factory, position)
    env.reset()
    _, _, terminated, truncated, info = env.step(np.zeros((3, 3)))
    assert terminated is failed
    assert truncated is (not failed)
    assert info["attacker_reached_base"] is failed
    assert info["distance_to_base"] == pytest.approx(np.linalg.norm(position))


@pytest.mark.parametrize("start,velocity,expected_failure", [
    (10.1, -2.0, True), (9.9, 2.0, False),
])
def test_attack_condition_uses_updated_state_not_previous_state(
    config_factory, start, velocity, expected_failure,
):
    env = PursuitEnv(config_factory(
        initialization={"attacker_position": [start, 0.0, 0.0],
                        "attacker_velocity": [velocity, 0.0, 0.0],
                        "pursuer_positions": [[-40, 0, 50], [-40, -20, 50], [-40, 20, 50]]},
        attacker=_quiet_aircraft(),
    ))
    env.reset()
    _, _, terminated, truncated, info = env.step(np.zeros((3, 3)))
    assert info["attacker_reached_base"] is expected_failure
    assert terminated is expected_failure
    assert not truncated
    assert info["distance_to_base"] == pytest.approx(start + velocity * env.dt)


@pytest.mark.parametrize("pursuer_x,pursuer_velocity,captured", [
    (15.0, 0.0, False),  # Capturable before the step, outside capture radius after it.
    (15.2, -4.0, True),  # Outside before the step, inside after synchronized motion.
])
def test_capture_and_attack_both_use_same_updated_instant(
    config_factory, pursuer_x, pursuer_velocity, captured,
):
    env = PursuitEnv(config_factory(
        environment={"max_episode_steps": 1},
        initialization={
            "attacker_position": [10.1, 0.0, 0.0],
            "attacker_velocity": [-2.0, 0.0, 0.0],
            "pursuer_positions": [[pursuer_x, 0, 0], [-40, -20, 50], [-40, 20, 50]],
            "pursuer_velocities": [[pursuer_velocity, 0, 0], [0, 0, 0], [0, 0, 0]],
        },
        attacker=_quiet_aircraft(),
    ))
    env.reset()
    _, _, terminated, truncated, info = env.step(np.zeros((3, 3)))
    assert terminated and not truncated
    assert info["distance_to_base"] == pytest.approx(9.9)
    assert info["attacker_in_attack_region"]
    assert info["captured"] is captured
    assert info["interception_success"] is captured
    assert info["attacker_reached_base"] is (not captured)
    assert info["reward_components"]["attack_failure"] == (0.0 if captured else -100.0)
    assert info["reward_components"]["capture"] == (100.0 if captured else 0.0)


def test_unopposed_attacker_closes_on_base_without_boundary_help(config_factory):
    env = PursuitEnv(config_factory(initialization={
        "pursuer_positions": [[-500, -100, 100], [-500, 0, 100], [-500, 100, 100]],
    }))
    _, initial = env.reset(seed=0)
    distances = [initial["distance_to_base"]]
    for _ in range(120):
        observation, reward, terminated, truncated, info = env.step(np.zeros((3, 3)))
        distances.append(info["distance_to_base"])
        assert np.isfinite(observation).all() and np.isfinite(reward).all()
        assert not terminated and not truncated
        assert info["boundary_clip_count"] == 0
    assert np.all(np.diff(distances) < 0.0)
    assert distances[-1] < distances[0] - 100.0


def test_legacy_pursuit_keeps_original_terminal_and_reward_semantics(legacy_config_factory):
    env = PursuitEnv(legacy_config_factory(environment={"max_episode_steps": 1}))
    assert not env.mission_enabled
    env.reset(seed=0)
    _, reward, terminated, truncated, info = env.step(np.zeros((3, 3)))
    assert not terminated and truncated
    assert not info["attacker_reached_base"]
    assert info["reward_components"].get("attack_failure", 0.0) == 0.0
    assert np.isfinite(reward).all()
