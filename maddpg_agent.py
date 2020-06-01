# main code that contains the neural network setup
# policy + critic updates
# see ddpg_agent.py for other details in the network

from ddpg_agent import DDPGAgent
from buffer import ReplayBuffer
import torch
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

GAMMA = 0.99                # discount factor
TAU = 0.001                 # soft update parameter
LR_ACTOR = 1e-4             # learning rate for updating actor
LR_CRITIC = 1e-3            # learning rate for updating critic
WEIGHT_DECAY = 0.0          # weight decay
BUFFER_SIZE = int(1e5)      # replay buffer size
BATCH_SIZE = 256            # mini batch size

class MADDPG:
    def __init__(self, num_agents, state_size, action_size, hidden_layers, 
                 seed, gamma = GAMMA, tau = TAU, lr_actor = LR_ACTOR, 
                 lr_critic = LR_CRITIC, weight_decay = WEIGHT_DECAY, 
                 buffer_size = BUFFER_SIZE, batch_size = BATCH_SIZE):
        """Initialize MADDPG agent."""
        super(MADDPG, self).__init__()

        self.seed = random.seed(seed)

        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.weight_decay = weight_decay
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.agents = [DDPGAgent(state_size, action_size, hidden_layers, gamma, \
                                 tau, lr_actor, lr_critic, weight_decay, seed) \
                                     for _ in range(num_agents)]

        self.replay_buffer = ReplayBuffer(num_agents, buffer_size, batch_size)

    def act(self, states):
        actions = np.zeros([self.num_agents, self.action_size])
        for index, agent in enumerate(self.agents):
            actions[index, :] = agent.act(states[index])
        return actions

    def step(self, states, actions, rewards, next_states, dones):
        """One step for MADDPG agent, include store the current transition and update parameters."""
        self.replay_buffer.add(states, actions, rewards, next_states, dones)

        if len(self.replay_buffer) > self.batch_size:
            '''
            experiences = self.replay_buffer.sample()
            states_list, _, _, _, _ = experiences
            next_actions_list = [self.agents[idx].target_actor(states).detach() \
                for idx, states in enumerate(states_list)]
            for i in range(self.num_agents):
                self.agents[i].step_learn(experiences, next_actions_list, i)
            '''
            for agent in self.agents:
                experiences = self.replay_buffer.sample()
                agent.step_learn(experiences)

    def save_weights(self):
        for index, agent in enumerate(self.agents):
            torch.save(agent.critic.state_dict(), 'agent{}_critic_trained_with_DDPG.pth'.format(index+1))
            torch.save(agent.actor.state_dict(), 'agent{}_actor_trained_with_DDPG.pth'.format(index+1))

    def reset(self):
        for agent in self.agents:
            agent.reset()
