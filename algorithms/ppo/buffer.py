"""On-policy storage with independent time and environment axes."""

from copy import deepcopy

import numpy as np


class VectorRolloutBuffer:
    """Keep each environment's reward, value and GAE trace independent.

    A true terminal includes capture, defense failure and the safety time limit.
    A partially collected rollout retains its nonterminal next-state bootstrap.
    State serialization includes unfinished samples and their old Gaussians.
    """

    def __init__(self, capacity, num_envs, observation_dim=16, action_dim=3):
        dimensions = (capacity, num_envs, observation_dim, action_dim)
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in dimensions):
            raise ValueError("Rollout dimensions must be positive integers")
        self.capacity, self.num_envs, self.observation_dim, self.action_dim = dimensions
        self.observations = np.zeros((capacity, num_envs, observation_dim), dtype=np.float32)
        for name in ("raw_actions", "means", "log_stds"):
            setattr(self, name, np.zeros((capacity, num_envs, action_dim), dtype=np.float32))
        for name in ("log_probs", "rewards", "values", "next_values", "advantages", "returns"):
            setattr(self, name, np.zeros((capacity, num_envs), dtype=np.float32))
        self.terminated = np.zeros((capacity, num_envs), dtype=bool)
        self.reset()

    @property
    def array_names(self):
        return ("observations", "raw_actions", "log_probs", "means", "log_stds",
                "rewards", "values", "next_values", "terminated", "advantages", "returns")

    def reset(self):
        self.size = 0
        self.returns_ready = False

    def add(self, observations, raw_actions, log_probs, means, log_stds,
            rewards, values, next_values, terminated):
        if self.size >= self.capacity:
            raise RuntimeError("Rollout buffer is full")
        incoming = dict(observations=observations, raw_actions=raw_actions,
                        log_probs=log_probs, means=means, log_stds=log_stds,
                        rewards=rewards, values=values, next_values=next_values)
        converted = {}
        for name, value in incoming.items():
            if np.iscomplexobj(value):
                raise ValueError(name + " must be real")
            array = np.asarray(value, dtype=np.float32)
            shape = getattr(self, name).shape[1:]
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(name + " must be finite with shape " + str(shape))
            converted[name] = array
        terminals = np.asarray(terminated)
        if terminals.shape != (self.num_envs,) or terminals.dtype != np.bool_:
            raise ValueError("terminated must be a boolean vector with one entry per environment")
        for name, array in converted.items():
            getattr(self, name)[self.size] = array
        self.terminated[self.size] = terminals
        self.size += 1
        self.returns_ready = False

    def compute_returns(self, gamma, gae_lambda):
        if self.size == 0:
            raise RuntimeError("Cannot compute returns for an empty rollout")
        if (not np.isfinite([gamma, gae_lambda]).all()
                or not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1):
            raise ValueError("gamma and gae_lambda must be finite and within [0,1]")
        advantage = np.zeros(self.num_envs, dtype=np.float64)
        for index in reversed(range(self.size)):
            alive = ~self.terminated[index]
            bootstrap = np.where(alive, self.next_values[index], 0).astype(np.float64)
            delta = self.rewards[index].astype(np.float64) + gamma * bootstrap - self.values[index]
            advantage = delta + gamma * gae_lambda * alive * advantage
            self.advantages[index] = advantage
            self.returns[index] = advantage + self.values[index]
        if not (np.isfinite(self.advantages[:self.size]).all()
                and np.isfinite(self.returns[:self.size]).all()):
            raise FloatingPointError("Non-finite vector GAE")
        self.returns_ready = True
        return self.advantages[:self.size].copy(), self.returns[:self.size].copy()

    def flattened(self):
        if not self.returns_ready or self.size == 0:
            raise RuntimeError("Compute GAE before flattening the rollout")
        return {name: getattr(self, name)[:self.size].reshape(
            self.size * self.num_envs, *getattr(self, name).shape[2:])
                for name in self.array_names}

    def state_dict(self):
        return dict(format_version=1, capacity=self.capacity, num_envs=self.num_envs,
                    observation_dim=self.observation_dim, action_dim=self.action_dim,
                    size=self.size, returns_ready=self.returns_ready,
                    arrays={name: getattr(self, name).copy() for name in self.array_names})

    def load_state_dict(self, state):
        state = deepcopy(state)
        for key in ("capacity", "num_envs", "observation_dim", "action_dim"):
            if state.get(key) != getattr(self, key):
                raise ValueError("Rollout checkpoint dimension differs: " + key)
        size = state.get("size")
        if (state.get("format_version") != 1 or isinstance(size, bool)
                or not isinstance(size, int) or not 0 <= size <= self.capacity
                or not isinstance(state.get("returns_ready"), bool)):
            raise ValueError("Invalid rollout checkpoint counters")
        arrays = state.get("arrays", {})
        if set(arrays) != set(self.array_names):
            raise ValueError("Rollout checkpoint array schema differs")
        for name, value in arrays.items():
            target = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError("Invalid rollout checkpoint array: " + name)
            if not np.isfinite(value).all():
                raise ValueError("Non-finite rollout checkpoint array: " + name)
        for name, value in arrays.items():
            getattr(self, name)[:] = value
        self.size, self.returns_ready = size, state["returns_ready"]
