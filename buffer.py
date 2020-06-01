import numpy as np
import random
from collections import namedtuple, deque
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer():
    def __init__(self, num_agents, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.

           Arguments
           ---------
           num_agents (int): Number of agents
           buffer_size (int): Maximum size of buffer
           batch_size (int): Size of each training batch
           seed (int): Random seed
        """
        self.num_agents = num_agents
        self.memory = deque(maxlen = buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names = \
            ["states", "actions", "rewards", "next_states", "dones"])

    def add(self, state, action, reward, next_action, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_action, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory.

           Return
           ------
           states_list: list (length of num_agents) of 2D tensors (batch_size * state_size)
           action_list: list (length of num_agents) of 2D tensors (batch_size * action_size)
           rewards: 2D tensors (batch_size * num_agents)
           next_states_list: list (length of num_agents) of 2D tensors (batch_size * state_size)
           dones: 2D tensors (batch_size * num_agents)
        """
        experiences = random.sample(self.memory, k = self.batch_size)

        states_list = [torch.from_numpy(np.vstack([e.states[idx] \
            for e in experiences if e is not None])).float().to(device) \
                for idx in range(self.num_agents)]
        actions_list = [torch.from_numpy(np.vstack([e.actions[idx] \
            for e in experiences if e is not None])).float().to(device) \
                for idx in range(self.num_agents)]
        rewards = torch.from_numpy(np.vstack([e.rewards \
            for e in experiences if e is not None])).float().to(device)
        next_states_list = [torch.from_numpy(np.vstack([e.next_states[idx] \
            for e in experiences if e is not None])).float().to(device) \
                for idx in range(self.num_agents)]
        dones = torch.from_numpy(np.vstack([e.dones \
            for e in experiences if e is not None])).long().to(device)

        return (states_list, actions_list, rewards, next_states_list, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)