"""Optional fixed input scaling, derived solely from physical configuration."""

import numpy as np


def build_input_scales(environment_config, normalization_config):
    """Return actor/critic divisors; disabled switches return all ones.

    No running statistics or clipping are introduced. Relative positions use
    the workspace extent; relative velocities use sums of the two aircraft's
    component speed limits. Absolute critic positions use maximum magnitudes.
    Scales are saved as network buffers and reused unchanged at evaluation.
    """
    environment = environment_config["environment"]
    low = np.asarray(environment["bounds"]["low"], dtype=np.float32)
    high = np.asarray(environment["bounds"]["high"], dtype=np.float32)

    def velocity_scale(limits):
        return np.array([limits["max_horizontal_speed"], limits["max_horizontal_speed"],
                         max(limits["max_climb_speed"], limits["max_descent_speed"])],
                        dtype=np.float32)

    pursuer = velocity_scale(environment_config["pursuer"])
    aircraft_key = "attacker" if "mission" in environment_config else "evader"
    evader = velocity_scale(environment_config[aircraft_key])
    extent = high - low
    observation = np.concatenate((pursuer, extent, pursuer + evader,
                                  np.tile(extent, 2), np.tile(pursuer * 2, 2)))
    position = np.maximum(np.abs(low), np.abs(high))
    state = np.concatenate((np.tile(np.concatenate((position, pursuer)), 3), position, evader))
    if not normalization_config["observation"]:
        observation = np.ones_like(observation)
    if not normalization_config["global_state"]:
        state = np.ones_like(state)
    return observation, state
