# individual network settings for each actor + critic pair
# see networkforall for details

from networkforall import Network
from torch.optim import Adam
import torch
import torch.nn.functional as F
import numpy as np

# add OU noise for exploration
from OUNoise import OUNoise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

class DDPGAgent:
    def __init__(self, state_size, action_size, hidden_layers, gamma,
                 tau, lr_actor, lr_critic, weight_decay, seed):
        """Initialize DDPG agent."""
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.actor = Network(state_size, hidden_layers[0], hidden_layers[1], 
                             action_size, seed, actor=True).to(device)
        self.critic = Network(2 * (state_size + action_size), hidden_layers[2], 
                              hidden_layers[3], 1, seed).to(device)
        self.target_actor = Network(state_size, hidden_layers[0], 
                                    hidden_layers[1], action_size, seed,
                                    actor=True).to(device)
        self.target_critic = Network(2 * (state_size + action_size), hidden_layers[2], 
                                     hidden_layers[3], 1, seed).to(device)

        self.noise = OUNoise(action_size, seed, scale = 1.0)

        '''
        # initialize targets same as original networks
        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)
        '''

        self.actor_optimizer = Adam(self.actor.parameters(), lr = lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr = lr_critic, 
                                     weight_decay = weight_decay)

    def act(self, state):
        """Calculate actions under current policy for a specific agent."""
        state = torch.from_numpy(state).float().to(device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        self.actor.train()
        action += self.noise.noise()
        return np.clip(action, -1, 1)

    def step_learn(self, experiences):
        """Update actor and critic using sampled experiences."""

        # states_list: list (length of num_agents) of 2D tensors (batch_size * state_size)
        # action_list: list (length of num_agents) of 2D tensors (batch_size * action_size)
        # rewards: 2D tensors (batch_size * num_agents)
        # next_states_list: list (length of num_agents) of 2D tensors (batch_size * state_size)
        # dones: 2D tensors (batch_size * num_agents)
        states_list, actions_list, rewards, next_states_list, dones = experiences

        next_full_states = torch.cat(next_states_list, dim = 1).to(device)   # 2D tensor (batch_size * (num_agents*state_size))
        full_states = torch.cat(states_list, dim = 1).to(device)             # 2D tensor (batch_size * (num_agents*state_size))
        full_actions = torch.cat(actions_list, dim = 1).to(device)           # 2D tensor (batch_size * (num_agents*action_size))

        # update critic
        next_actions_list = [self.target_actor(states) for states in states_list]
        next_full_actions = torch.cat(next_actions_list, dim = 1).to(device)
        Q_target_next = self.target_critic(next_full_states, next_full_actions)   # 2D tensor (batch_size * 1)
        '''
        Q_target = rewards[:, idx_agent].view(-1, 1) + \
            self.gamma * Q_target_next * (1.0 - dones[:, idx_agent].view(-1, 1))
        '''
        Q_target = rewards + (self.gamma * Q_target_next * (1.0 - dones))
        Q_predict = self.critic(full_states, full_actions)                        # 2D tensor (batch_size * 1)
        critic_loss = F.mse_loss(Q_predict, Q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update actor
        '''
        action_pred = self.actor(states_list[idx_agent])     # 2D tensor (batch_size * action_size)
        actions_list_update = actions_list.copy()
        actions_list_update[idx_agent] = action_pred
        full_actions_update = torch.cat(actions_list_update, dim = 1).to(device)   # 2D tensor (batch_size * (num_agents*action_size))
        '''
        actions_pred = [self.actor(states) for states in states_list]        
        actions_pred_tensor = torch.cat(actions_pred, dim=1).to(device)
        # actor_loss = -self.critic(full_states, full_actions_update).mean()
        actor_loss = -self.critic(full_states, actions_pred_tensor).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # soft update target networks
        self.soft_update(self.target_critic, self.critic, self.tau)
        self.soft_update(self.target_actor, self.actor, self.tau)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def reset(self):
        self.noise.reset()