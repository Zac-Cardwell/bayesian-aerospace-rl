import torch 
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import torch.nn.functional as F
import numpy as np


class RunningMeanStd:
    """Tracks running mean and std for observation normalization."""
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon

    def update(self, x):
        """Update running statistics with a batch of observations."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count

    def normalize(self, x, clip=10.0):
        """Normalize observations using running statistics."""
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip)


class Critic(nn.Module):
    """
    Critic network that estimates Q(s,a) values.
    
    Args:
        in_dim: Input dimension (state_dim + action_dim)
        hidden_dim: Hidden layer dimension
    
    Input shape: (batch_size, state_dim + action_dim)
    Output shape: (batch_size, 1)
    """
    def __init__(self, in_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        out = torch.relu(self.fc1(state))
        return self.fc2(out)
    

class Actor(nn.Module):
    """
    Actor network that outputs a stochastic policy.
    
    Args:
        state_dim: State space dimension
        action_dim: Action space dimension
        hidden_dim: Hidden layer dimension
    
    Input shape: (batch_size, state_dim)
    Output shapes:
        - action: (batch_size, action_dim)
        - log_prob: (batch_size, 1)
        - mean: (batch_size, action_dim)
        - log_std: (batch_size, action_dim)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))

        mean = self.mean(x)
        log_std = self.log_std(x).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()  
        action = torch.tanh(z)
        
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob, mean, log_std

       
    



class ReplayBuffer:
    """
    Stores transitions as (state, action, reward, next_state, done) tuples.
    All elements should be 1D tensors (no batch dimension):
        - state: (state_dim,)
        - action: (action_dim,)
        - reward: (1,) or scalar
        - next_state: (state_dim,)
        - done: (1,) or scalar
    """
    def __init__(self, buffer_size=10000, device=None):
        self.buffer = deque(maxlen=buffer_size)
        self.device = device

    def size(self):
        return len(self.buffer)
    
    def add(self, experience):
        state, action, reward, next_state, done = experience

        # Convert everything to numpy
        state = np.asarray(state, dtype=np.float32).squeeze()
        next_state = np.asarray(next_state, dtype=np.float32).squeeze()
        action = np.asarray(action, dtype=np.float32).squeeze()

        reward = np.asarray(reward, dtype=np.float32).reshape(1)
        done = np.asarray(done, dtype=np.float32).reshape(1)

        self.buffer.append((state, action, reward, next_state, done))

        
    def sample(self, batch_size=64):
        """
        Sample batch and stack into batched tensors.
        
        Returns:
            Tuple of batched tensors:
                - states: (batch_size, state_dim)
                - actions: (batch_size, action_dim)
                - rewards: (batch_size, 1)
                - next_states: (batch_size, state_dim)
                - dones: (batch_size, 1)
        """
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))

        device = self.device 

        states = torch.tensor(states, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.float32, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)

        return states, actions, rewards, next_states, dones



class SAC(nn.Module):
    """
    Soft Actor-Critic (SAC) algorithm implementation.
    
    SAC is an off-policy actor-critic deep RL algorithm that optimizes
    a stochastic policy with maximum entropy reinforcement learning.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dim: Hidden layer dimension for networks
        actor_lr: Learning rate for actor network
        critic_lr: Learning rate for critic networks
        gamma: Discount factor
        buffer_size: Maximum size of replay buffer
        batch_size: Batch size for training
        actor_update_freq: how often the actor is updated 
        tau: Soft update coefficient for target networks
        max_action: Maximum absolute value of actions
        lr_schedule: shoul,d the model use a lr scheduler
        normalize_obs: Whether to normalize observations
        device: Device to run on (cuda/cpu)
        ```
    """
    def __init__(self, state_dim, action_dim, hidden_dim, actor_lr=3e-4, 
                 critic_lr=3e-4, alpha_lr=3e-4, gamma=0.99, buffer_size=10000, batch_size=128, 
                 actor_update_freq=2, tau=0.005, max_action=1.0, lr_schedule=None, normalize_obs=False, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.gamma = gamma
        self.memory = ReplayBuffer(buffer_size=buffer_size, device=self.device)
        self.batch_size = batch_size
        self.actor_update_freq = actor_update_freq
        self.update_counter = 0
        self.tau = tau
        self.max_action = max_action
        self.normalize_obs = normalize_obs
        self.training_mode = True

        # Observation normalization
        if self.normalize_obs:
            self.obs_rms = RunningMeanStd(shape=(state_dim,))
        
        # Automatic entropy tuning
        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = -float(action_dim)

        # Actor network
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Twin critic networks
        self.critic1 = Critic(state_dim + action_dim, hidden_dim).to(self.device)
        self.critic2 = Critic(state_dim + action_dim, hidden_dim).to(self.device)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        
        # Target critic networks
        self.critic1_target = Critic(state_dim + action_dim, hidden_dim).to(self.device)
        self.critic2_target = Critic(state_dim + action_dim, hidden_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        if lr_schedule == 'linear':
            self.actor_scheduler = optim.lr_scheduler.LinearLR(
                self.actor_optimizer, start_factor=1.0, end_factor=0.1, total_iters=1000000
            )
            self.critic_schedulers = [
                optim.lr_scheduler.LinearLR(self.critic1_optimizer, start_factor=1.0, end_factor=0.1, total_iters=1000000),
                optim.lr_scheduler.LinearLR(self.critic2_optimizer, start_factor=1.0, end_factor=0.1, total_iters=1000000)
            ]
        else:
            self.actor_scheduler = None
            self.critic_schedulers = None

    def normalize_state(self, state):
        """
        Normalize state using running statistics.
        
        Args:
            state: State tensor or numpy array
        
        Returns:
            Normalized state as tensor
        """
        if not self.normalize_obs:
            return state
        
        if isinstance(state, torch.Tensor):
            state_np = state.cpu().numpy()
            normalized = self.obs_rms.normalize(state_np)
            return  torch.tensor(normalized, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            normalized = self.obs_rms.normalize(state)
            return  torch.tensor(normalized, dtype=torch.float32, device=self.device).unsqueeze(0)
        
    def step_schedulers(self):
        """Call this after update() to step LR schedulers."""
        if self.actor_scheduler:
            self.actor_scheduler.step()
        if self.critic_schedulers:
            for scheduler in self.critic_schedulers:
                scheduler.step()

    def update_obs_stats(self, states):
        """
        Update observation normalization statistics.
        
        Args:
            states: Batch of states, Shape: (batch_size, state_dim)
        """
        if self.normalize_obs:
            if isinstance(states, torch.Tensor):
                states = states.cpu().numpy()
            self.obs_rms.update(states)

    def soft_update(self):
        """Soft update of target network parameters using Polyak averaging."""
        for target_param, param in zip(self.critic1_target.parameters(), 
                                       self.critic1.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

        for target_param, param in zip(self.critic2_target.parameters(), 
                                       self.critic2.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def get_action(self, state, deterministic=False):
        if self.normalize_obs:
            state = self.normalize_state(state)

        if isinstance(state, (list, np.ndarray)):
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            
        with torch.no_grad():
            action, log_prob, mean, log_std = self.actor(state)
            if deterministic:
                return mean * self.max_action
            return action * self.max_action

    def train_mode(self):
        self.training_mode = True
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval_mode(self):
        self.training_mode = False
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

    def forward(self, state):
        return self.actor(state)
    
    def update(self, soft=True):
        if self.memory.size() < self.batch_size:
            return None, None, None
        
        self.update_counter += 1

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Update observation statistics
        if self.normalize_obs:
            self.update_obs_stats(states)
            states = self.normalize_state(states)
            next_states = self.normalize_state(next_states)
        
        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            next_actions, next_log_prob, _, _ = self.actor(next_states)
            q1_next = self.critic1_target(torch.cat([next_states, next_actions], dim=1))
            q2_next = self.critic2_target(torch.cat([next_states, next_actions], dim=1))
            q_next = torch.min(q1_next, q2_next)

            target_q = rewards + self.gamma * (1 - dones) * (q_next - alpha * next_log_prob)

        current_q1 = self.critic1(torch.cat([states, actions], dim=1))
        current_q2 = self.critic2(torch.cat([states, actions], dim=1))

        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)
        self.critic2_optimizer.step()

        actor_loss = None
        if self.update_counter % self.actor_update_freq == 0:
            new_actions, new_log_prob, _, _ = self.actor(states)

            q1_new = self.critic1(torch.cat([states, new_actions], dim=1))
            q2_new = self.critic2(torch.cat([states, new_actions], dim=1))
            q_new = torch.min(q1_new, q2_new)

            actor_loss = (alpha * new_log_prob - q_new).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            alpha_loss = -(self.log_alpha * (new_log_prob + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        if soft:
            self.soft_update()


        actor_loss_value = None if actor_loss is None else actor_loss.item()
        return actor_loss_value, critic1_loss.item(), critic2_loss.item()
