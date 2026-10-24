"""Direct guidance tracks velocity; the shared dynamics owns all limits."""

from copy import deepcopy

import numpy as np
import pytest

from envs import PursuitEnv
from envs.attacker_policy import DirectAttackerPolicy, GoalDirectedAttackerPolicy
from envs.dynamics import MotionLimits, integrate


@pytest.fixture
def direct_policy(baseline_config):
    policy = baseline_config["attacker"]["policy"]
    return DirectAttackerPolicy(
        base_position=baseline_config["mission"]["base_position"],
        attack_speed=policy["attack_speed"], velocity_gain=policy["velocity_gain"],
        distance_epsilon=policy["distance_epsilon"],
    )


def _state(position, velocity=(0.0, 0.0, 0.0)):
    return np.concatenate((np.asarray(position, dtype=np.float64), velocity))


@pytest.mark.parametrize("position", [
    [500.0, 0.0, 50.0], [-100.0, 20.0, -40.0], [0.0, 0.0, -30.0],
])
def test_desired_velocity_always_points_towards_configured_base(position):
    base = np.array([3.0, -5.0, 7.0])
    policy = DirectAttackerPolicy(
        base_position=base, attack_speed=17.0, velocity_gain=0.8, distance_epsilon=1e-6,
    )
    state = _state(position, [5.0, -3.0, 2.0])
    goal = base - state[:3]
    desired = policy.desired_velocity(state)
    np.testing.assert_allclose(desired, 17.0 * goal / np.linalg.norm(goal))
    assert np.linalg.norm(desired) == pytest.approx(17.0)
    assert np.dot(desired, goal) > 0.0
    assert desired.shape == (3,) and np.isfinite(desired).all()


def test_command_corrects_lateral_velocity_and_tracks_the_requested_gain():
    policy = DirectAttackerPolicy(
        base_position=[0.0, 0.0, 0.0], attack_speed=20.0,
        velocity_gain=0.75, distance_epsilon=1e-6,
    )
    state = _state([100.0, 0.0, 0.0], [0.0, 8.0, -3.0])
    command = policy(state)
    np.testing.assert_allclose(command, [-15.0, -6.0, 2.25])
    assert command[1] * state[4] < 0.0
    assert command[2] * state[5] < 0.0
    assert np.isfinite(command).all()


def test_pursuer_positions_and_velocities_do_not_change_direct_guidance(direct_policy):
    state = _state([30.0, 4.0, 2.0], [-5.0, 2.0, 1.0])
    coincident = np.tile(state, (3, 1))
    nearby = coincident + np.array([0.01, -0.02, 0.03, 2.0, -3.0, 4.0])
    distant = coincident + np.array([500.0, 500.0, 500.0, -10.0, 10.0, 0.0])
    reference = direct_policy(state)
    for pursuers in (None, coincident, nearby, distant):
        np.testing.assert_array_equal(direct_policy(state, pursuers), reference)
    assert not direct_policy.boundary_enabled


def test_policy_returns_raw_physical_error_and_dynamics_applies_existing_limits(
    direct_policy, baseline_config,
):
    state = _state([300.0, 0.0, 100.0], [0.0, 5.0, 1.0])
    original = state.copy()
    limits = MotionLimits.from_config(baseline_config["attacker"])
    dt = baseline_config["environment"]["dt"]
    command = direct_policy(state)
    expected = direct_policy.velocity_gain * (direct_policy.desired_velocity(state) - state[3:])
    np.testing.assert_allclose(command, expected)
    # A bounded or normalized policy output would conceal the velocity error.
    assert np.linalg.norm(command[:2]) > limits.max_horizontal_acceleration
    assert abs(command[2]) > limits.max_vertical_acceleration
    next_state, applied = integrate(state, command, np.zeros(3), dt, limits)
    assert np.linalg.norm(applied[:2]) <= limits.max_horizontal_acceleration + 1e-12
    assert abs(applied[2]) <= limits.max_vertical_acceleration + 1e-12
    assert np.linalg.norm(applied) <= limits.max_jerk * dt + 1e-12
    assert np.linalg.norm(next_state[3:5]) <= limits.max_horizontal_speed + 1e-12
    assert -limits.max_descent_speed <= next_state[5] <= limits.max_climb_speed
    assert np.isfinite(next_state).all() and np.isfinite(applied).all()
    np.testing.assert_array_equal(state, original)


@pytest.mark.parametrize("distance", [500.0, 100.0, 10.01, 9.99])
def test_guidance_does_not_brake_on_approach_or_inside_attack_sphere(direct_policy, distance):
    desired = direct_policy.desired_velocity(_state([distance, 0.0, 0.0]))
    np.testing.assert_allclose(desired, [-direct_policy.attack_speed, 0.0, 0.0])
    # Mission termination, rather than an arrival controller, ends flight.
    tracked = _state([distance, 0.0, 0.0], desired)
    np.testing.assert_allclose(direct_policy(tracked), np.zeros(3), atol=1e-12)


@pytest.mark.parametrize("offset", [0.0, 1e-12, 1e-6])
def test_zero_distance_guard_is_finite_and_uses_zero_desired_velocity(direct_policy, offset):
    state = _state([offset, 0.0, 0.0], [1.0, -2.0, 3.0])
    np.testing.assert_array_equal(direct_policy.desired_velocity(state), np.zeros(3))
    np.testing.assert_allclose(direct_policy(state), -direct_policy.velocity_gain * state[3:])
    assert np.isfinite(direct_policy(state)).all()


def test_unopposed_full_500m_flight_enters_strict_attack_sphere_without_orbit_or_clip(
    direct_policy, baseline_config,
):
    config = baseline_config
    mission = config["mission"]
    initial = config["initialization"]
    state = _state(initial["attacker_position"], initial["attacker_velocity"])
    acceleration = np.zeros(3)
    limits = MotionLimits.from_config(config["attacker"])
    dt = config["environment"]["dt"]
    low = np.asarray(config["environment"]["bounds"]["low"])
    high = np.asarray(config["environment"]["bounds"]["high"])
    base = np.asarray(mission["base_position"])
    radius = mission["attack_radius"]
    distances = [np.linalg.norm(state[:3] - base)]
    assert distances[0] == pytest.approx(mission["detection_radius"])
    reached = False
    for step in range(1, config["environment"]["max_episode_steps"] + 1):
        state, acceleration = integrate(state, direct_policy(state), acceleration, dt, limits)
        assert np.isfinite(state).all() and np.isfinite(acceleration).all()
        assert np.all(state[:3] >= low) and np.all(state[:3] <= high)
        distances.append(np.linalg.norm(state[:3] - base))
        if distances[-1] < radius:
            reached = True
            break
    assert reached, "Direct attacker must reach the existing 10 m sphere before timeout"
    assert np.all(np.diff(distances) < 0.0), "Approach must not reverse into orbit or oscillation"
    assert distances[-2] >= radius > distances[-1]
    entry_speed = np.linalg.norm(state[3:])
    assert entry_speed >= 0.8 * direct_policy.attack_speed
    maximum_3d_speed = np.hypot(
        limits.max_horizontal_speed, max(limits.max_climb_speed, limits.max_descent_speed)
    )
    assert (distances[0] - radius) / maximum_3d_speed <= step * dt
    assert step * dt < config["environment"]["max_episode_steps"] * dt


def test_direct_environment_ends_on_attack_and_preserves_all_policy_inputs(config_factory):
    # Far, stationary interceptors isolate the attack outcome without changing
    # the attacker policy, task thresholds, timing or physical limits.
    env = PursuitEnv(config_factory(initialization={
        "pursuer_positions": [[-500, -100, 100], [-500, 0, 100], [-500, 100, 100]],
    }))
    observation, info = env.reset(seed=0)
    assert isinstance(env.attacker_policy, DirectAttackerPolicy)
    previous_distance = info["distance_to_base"]
    for _ in range(env.max_episode_steps):
        observation, reward, terminated, truncated, info = env.step(np.zeros((3, 3)))
        assert observation.shape == (3, 21)
        assert env.get_state().shape == info["global_state"].shape == (24,)
        assert np.isfinite(observation).all() and np.isfinite(reward).all()
        assert np.isfinite(info["global_state"]).all()
        assert info["boundary_clip_count"] == 0
        assert info["distance_to_base"] < previous_distance
        previous_distance = info["distance_to_base"]
        if terminated or truncated:
            break
    assert terminated and not truncated
    assert info["enemy_attack_success"] and info["attacker_reached_base"]
    assert not info["interception_success"] and not info["captured"]
    assert info["distance_to_base"] < env.attack_radius
    assert info["reward_components"]["attack_failure"] == env.reward_coefficients["attack_failure_penalty"]


def test_default_yaml_selects_direct_mode_without_avoidance_or_boundary(baseline_config):
    parameters = baseline_config["attacker"]["policy"]
    assert parameters["mode"] == "direct"
    assert parameters["attack_speed"] == baseline_config["attacker"]["max_horizontal_speed"]
    assert parameters["velocity_gain"] > 0.0
    assert "goal_weight" not in parameters and "avoidance_weight" not in parameters
    assert not any(name.startswith("boundary_") for name in parameters)


def test_direct_guidance_parameters_are_consumed_from_yaml(config_factory, baseline_config):
    parameters = deepcopy(baseline_config["attacker"]["policy"])
    parameters.update(attack_speed=13.0, velocity_gain=0.6, distance_epsilon=1e-5)
    env = PursuitEnv(config_factory(attacker={"policy": parameters}))
    state = _state([100.0, 0.0, 0.0], [0.0, 2.0, 0.0])
    np.testing.assert_allclose(env.attacker_policy.desired_velocity(state), [-13.0, 0.0, 0.0])
    np.testing.assert_allclose(env.attacker_policy(state), [-7.8, -1.2, 0.0])
    assert env.attacker_policy.distance_epsilon == 1e-5


@pytest.mark.parametrize("mode", ["unknown", "evasive", None, 1])
def test_unknown_explicit_attacker_mode_is_rejected(config_factory, baseline_config, mode):
    parameters = deepcopy(baseline_config["attacker"]["policy"])
    parameters["mode"] = mode
    with pytest.raises(ValueError):
        PursuitEnv(config_factory(attacker={"policy": parameters}))


@pytest.mark.parametrize("boundary", [
    {"boundary_strength": 2.0},
    {"boundary_margin_horizontal": 50.0, "boundary_margin_vertical": 20.0,
     "boundary_strength": 2.0},
])
def test_direct_mode_rejects_boundary_steering_settings(config_factory, baseline_config, boundary):
    parameters = deepcopy(baseline_config["attacker"]["policy"])
    parameters.update(boundary)
    with pytest.raises(ValueError):
        PursuitEnv(config_factory(attacker={"policy": parameters}))


def test_historical_goal_avoidance_snapshot_keeps_its_original_policy(config_factory):
    env = PursuitEnv(config_factory(attacker={"policy": {
        "distance_epsilon": 1e-6, "goal_weight": 1.0, "avoidance_weight": 25.0,
    }}))
    assert isinstance(env.attacker_policy, GoalDirectedAttackerPolicy)
    env.reset(seed=0)
    _, reward, _, _, info = env.step(np.zeros((3, 3)))
    assert np.isfinite(reward).all() and np.isfinite(info["global_state"]).all()


@pytest.mark.parametrize("key,value", [
    ("attack_speed", 0.0), ("attack_speed", -1.0), ("attack_speed", np.nan),
    ("attack_speed", np.inf), ("attack_speed", 1j),
    ("velocity_gain", 0.0), ("velocity_gain", -1.0), ("velocity_gain", np.nan),
    ("velocity_gain", np.inf), ("velocity_gain", "bad"),
    ("distance_epsilon", 0.0), ("distance_epsilon", np.nan),
    ("distance_epsilon", np.inf), ("distance_epsilon", 1j),
    ("base_position", [0.0, 0.0]), ("base_position", [0.0, 0.0, np.nan]),
    ("base_position", [0.0, 0.0, np.inf]), ("base_position", [0.0, 0.0, 1j]),
])
def test_invalid_direct_guidance_configuration_is_rejected(key, value):
    parameters = dict(base_position=[0.0, 0.0, 0.0], attack_speed=20.0,
                      velocity_gain=1.0, distance_epsilon=1e-6)
    parameters[key] = value
    with pytest.raises(ValueError):
        DirectAttackerPolicy(**parameters)


@pytest.mark.parametrize("state", [
    np.zeros(3), np.zeros((1, 6)), np.zeros(6, dtype=complex),
    [0.0, 0.0, 0.0, np.nan, 0.0, 0.0], [np.inf, 0.0, 0.0, 0.0, 0.0, 0.0],
])
def test_invalid_attacker_states_are_rejected(direct_policy, state):
    with pytest.raises(ValueError):
        direct_policy.desired_velocity(state)
    with pytest.raises(ValueError):
        direct_policy(state)
