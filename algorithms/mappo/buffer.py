"""A single-environment, multi-pursuer on-policy rollout buffer."""

import numpy as np


class RolloutBuffer:
    """Store team timesteps, keeping actor and critic batch dimensions distinct.

    Observations/raw actions/log probabilities have an agent axis; rewards and
    values are stored once per team timestep, avoiding triple-counted returns.
    Each transition stores its next value before any environment reset.
    """

    def __init__(self, capacity, num_agents, observation_dim, state_dim, action_dim):
        dimensions = (capacity, num_agents, observation_dim, state_dim, action_dim)
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in dimensions):
            raise ValueError("Buffer dimensions must be positive integers")
        self.capacity = capacity
        self.num_agents = num_agents
        self.observations = np.zeros((capacity, num_agents, observation_dim), dtype=np.float32)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.raw_actions = np.zeros((capacity, num_agents, action_dim), dtype=np.float32)
        self.log_probs = np.zeros((capacity, num_agents), dtype=np.float32)
        for name in ("rewards", "values", "next_values", "advantages", "returns"):
            setattr(self, name, np.zeros(capacity, dtype=np.float32))
        self.terminated = np.zeros(capacity, dtype=bool)
        self.truncated = np.zeros(capacity, dtype=bool)
        self.reset()

    def reset(self):
        self.size = 0
        self.returns_ready = False

    def add(self, observations, state, raw_actions, log_probs, reward, value,
            next_value, terminated, truncated):
        if self.size >= self.capacity:
            raise RuntimeError("Rollout buffer is full")
        arrays = {
            "observations": observations, "states": state,
            "raw_actions": raw_actions, "log_probs": log_probs,
            "rewards": reward, "values": value, "next_values": next_value,
        }
        converted = {}
        for name, data in arrays.items():
            target = getattr(self, name)
            array = np.asarray(data, dtype=np.float32)
            if array.shape != target.shape[1:] or not np.isfinite(array).all():
                raise ValueError("{} must be finite with shape {}".format(name, target.shape[1:]))
            converted[name] = array
        for name, array in converted.items():
            getattr(self, name)[self.size] = array
        self.terminated[self.size] = bool(terminated)
        self.truncated[self.size] = bool(truncated)
        self.size += 1
        self.returns_ready = False

    def compute_returns(self, gamma, gae_lambda):
        """GAE with separate bootstrap and episode-continuation masks.

        Capture: no bootstrap. Timeout: bootstrap the final observed state but
        stop the trace before the next episode. A rollout cut in a continuing
        episode bootstraps its next state without inventing a terminal event.
        """
        if not self.size:
            raise RuntimeError("Cannot compute returns for an empty rollout")
        if not (np.isfinite([gamma, gae_lambda]).all()
                and 0 <= gamma <= 1 and 0 <= gae_lambda <= 1):
            raise ValueError("gamma and gae_lambda must lie in [0, 1]")
        advantage = 0.0
        for index in reversed(range(self.size)):
            bootstrap = 0.0 if self.terminated[index] else self.next_values[index]
            delta = self.rewards[index] + gamma * bootstrap - self.values[index]
            continues = not (self.terminated[index] or self.truncated[index])
            advantage = delta + gamma * gae_lambda * float(continues) * advantage
            self.advantages[index] = advantage
            self.returns[index] = advantage + self.values[index]
        if not (np.isfinite(self.advantages[:self.size]).all()
                and np.isfinite(self.returns[:self.size]).all()):
            raise FloatingPointError("Non-finite GAE output")
        self.returns_ready = True
        return self.advantages[:self.size].copy(), self.returns[:self.size].copy()
