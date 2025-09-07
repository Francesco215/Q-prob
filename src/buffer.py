# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# ---
# Adapted for 1-step Q-learning

from collections import deque
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, obs_shape: tuple[int, ...], action_dim: int, max_action: float, pixel_obs: bool,
                 device: torch.device, history: int=1, max_size: int=1e6, batch_size: int=256,
                 prioritized: bool=True, initial_priority: float=1, normalize_actions: bool=True):

        self.max_size = int(max_size)
        self.batch_size = batch_size
        self.device = device
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.history = history
        self.prioritized = prioritized
        self.action_scale = max_action if normalize_actions else 1.
        self.obs_dtype = torch.uint8 if pixel_obs else torch.float
        
        # Determine storage device (GPU or CPU)
        self.storage_device = torch.device('cpu')
        if self.device.type == 'cuda':
            # A rough estimate of memory usage to see if it fits in VRAM
            memory, _ = torch.cuda.mem_get_info()
            obs_space = np.prod((self.max_size, *self.obs_shape)) * (1 if pixel_obs else 4)
            other_space = self.max_size * (action_dim + 2) * 4 # action, reward, not_done
            if obs_space + other_space < memory * 0.8: # Use 80% of VRAM for safety
                self.storage_device = self.device

        # State shape includes history (e.g., stacked frames)
        self.state_shape = [obs_shape[0] * history] + list(obs_shape[1:])
        self.num_channels = obs_shape[0]

        # Core storage
        self.obs = torch.zeros((self.max_size, *self.obs_shape), device=self.storage_device, dtype=self.obs_dtype)
        # Store action, reward, and not_done flag together
        self.action = torch.zeros((self.max_size, self.action_dim), device=self.device)
        self.reward = torch.zeros((self.max_size, 1), device=self.device)
        self.not_done = torch.zeros((self.max_size, 1), device=self.device)

        # Efficiently track indices for states with history
        self.state_indices = np.zeros((self.max_size, self.history), dtype=np.int32)
        self.history_queue = deque(maxlen=self.history)
        for _ in range(self.history):
            self.history_queue.append(0)

        # Prioritization
        self.priority = torch.empty(self.max_size, device=self.device) if self.prioritized else None
        self.max_priority = initial_priority
        self.sampled_indices = None # To store indices for priority updates

        # Tracking
        self.ind = 0
        self.size = 0
        self.can_sample = torch.zeros(self.max_size, dtype=torch.bool)
        self.env_terminates = False

    def add(self, state: np.array, action: int | float, next_state: np.array, reward: float, terminated: bool, truncated: bool):
        # The current state is represented by the most recent frame/observation
        current_obs = torch.as_tensor(state[-self.num_channels:].reshape(self.obs_shape), 
                                      dtype=self.obs_dtype, device=self.storage_device)
        self.obs[self.ind] = current_obs

        # Store action, reward, done flag
        if isinstance(action, int):
            # Convert discrete action to one-hot
            one_hot_action = torch.zeros(self.action_dim, device=self.device)
            one_hot_action[action] = 1
            self.action[self.ind] = one_hot_action
        else:
            # Normalize continuous action
            self.action[self.ind] = torch.as_tensor(action / self.action_scale, dtype=torch.float, device=self.device)

        self.reward[self.ind] = reward
        self.not_done[self.ind] = 1.0 - terminated # `not_done` is important for the Bellman update
        if terminated: self.env_terminates = True
        
        # Store the indices that make up the current state (s_t)
        self.state_indices[self.ind] = np.array(self.history_queue, dtype=np.int32)
        
        # An index can be sampled if it and the following `history` frames are valid
        if self.size >= self.history:
            self.can_sample[self.ind - (self.history - 1)] = True

        if self.prioritized:
            self.priority[self.ind] = self.max_priority
        
        # Update history queue for the *next* state
        self.history_queue.append((self.ind + 1) % self.max_size)

        self.ind = (self.ind + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        if terminated or truncated:
            # When an episode ends, reset the history queue to avoid linking states across episodes
            self.obs[(self.ind) % self.max_size] = torch.as_tensor(
                next_state[-self.num_channels:].reshape(self.obs_shape), 
                dtype=self.obs_dtype, device=self.storage_device
            )
            # Invalidate samples that would cross the episode boundary
            self.can_sample[self.ind - (self.history - 1) : self.ind] = False
            for _ in range(self.history):
                self.history_queue.append((self.ind) % self.max_size)

    def _get_states(self, indices: np.ndarray) -> torch.Tensor:
        # Retrieve observations using the stored indices and reshape to form the state
        state_obs = self.obs[self.state_indices[indices]].to(self.device).float()
        return state_obs.reshape(self.batch_size, *self.state_shape)

    def _get_next_states(self, indices: np.ndarray) -> torch.Tensor:
        # The next state is formed from the history at s_t plus the new observation at s_{t+1}
        next_indices = (self.state_indices[indices][:, 1:] + 1) % self.max_size
        next_obs_index = (indices + 1) % self.max_size
        next_state_indices = np.concatenate([next_indices, next_obs_index[:, np.newaxis]], axis=1)
        
        next_state_obs = self.obs[next_state_indices].to(self.device).float()
        return next_state_obs.reshape(self.batch_size, *self.state_shape)

    def sample(self):
        # Determine valid indices for sampling
        valid_indices = torch.where(self.can_sample)[0]
        
        if self.prioritized:
            # Sample using priorities
            valid_priorities = self.priority[valid_indices]
            csum = torch.cumsum(valid_priorities, 0)
            rand_vals = torch.rand(size=(self.batch_size,), device=self.device) * csum[-1]
            sampled_relative_indices = torch.searchsorted(csum, rand_vals)
        else:
            # Uniform sampling
            sampled_relative_indices = torch.randint(0, len(valid_indices), size=(self.batch_size,), device=self.device)

        self.sampled_indices = valid_indices[sampled_relative_indices.cpu()].numpy()
        
        # Retrieve batch data
        state = self._get_states(self.sampled_indices)
        next_state = self._get_next_states(self.sampled_indices)
        action = self.action[self.sampled_indices]
        reward = self.reward[self.sampled_indices]
        not_done = self.not_done[self.sampled_indices]

        # For pixel observations, normalize to [0, 1]
        if self.obs_dtype == torch.uint8:
            state, next_state = state / 255.0, next_state / 255.0

        return state, action, next_state, reward, not_done

    @torch.no_grad()
    def update_priority(self, priority: torch.Tensor):
        if not self.prioritized: return
        self.priority[self.sampled_indices] = priority.reshape(-1).detach()
        self.max_priority = max(float(priority.max()), self.max_priority)