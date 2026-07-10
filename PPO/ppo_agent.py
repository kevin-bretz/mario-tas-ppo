import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from PPO import ppo_config

class PPONetwork(nn.Module):
    def __init__(self, state_size, hidden_size, action_size):
        super(PPONetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(state_size[0], 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *state_size)
            cnn_out = self.cnn(dummy)
            conv_out_size = cnn_out.shape[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return self.policy_head(x), self.value_head(x)

class PPOAgent:
    def __init__(
        self,
        env: gym.Env,
        state_size: int,
        hidden_size: int,
        action_size: int,
        learning_rate: float,
        gamma: float,
        epsilon: float,
        entropy_coef: float,
        batch_size: int,
        num_epochs: int,
        num_stack: int = 4,
        device=None,
        entropy_coef_min: float = 0.001,
        entropy_coef_decay: float = 0.995,
    ):
        self.env = env
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_stack = num_stack
        self.network = PPONetwork(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        self.entropy_coef = entropy_coef
        self.entropy_coef_min = entropy_coef_min
        self.entropy_coef_decay = entropy_coef_decay

        self.batch_size = batch_size
        self.num_epochs = num_epochs

        # Simple lists to hold transitions
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def select_action(self, state):
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device) / 255.0
        logits, value = self.network(state_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def select_action_sample(self, rgb_array):
        action, _, _ = self.select_action(rgb_array)
        return action

    def add_experience(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def compute_gae(self, rewards, values, next_values, dones, tau=0.95):
        rewards = rewards.view(-1)
        values = values.view(-1)
        next_values = next_values.view(-1)
        dones = dones.view(-1)

        advantages = torch.zeros_like(rewards)
        last_gae_lambda = 0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            advantages[t] = last_gae_lambda = (
                delta + self.gamma * tau * (1 - dones[t]) * last_gae_lambda
            )
        return advantages

    def set_entropy_coef(self, value):
        self.entropy_coef = value

    def decay_entropy_coef(self):
        # Multiplicatively decay the entropy bonus toward its floor, called once per episode.
        self.entropy_coef = max(self.entropy_coef_min, self.entropy_coef * self.entropy_coef_decay)
        return self.entropy_coef

    def update(self):
        if len(self.states) < 2:
            return 0.0, 0.0, 0.0

        for idx, s in enumerate(self.next_states):
            if s.shape != self.next_states[0].shape:
                print(f"Shape mismatch at index {idx}: {s.shape} vs {self.next_states[0].shape}")

        # Convert list of np arrays to a single np array, then to tensor
        states_np = np.stack(self.states)  # shape: (batch, 1, 84, 84)
        print(states_np.shape)
        states_t = torch.tensor(states_np, dtype=torch.float32).to(self.device) / 255.0

        actions_t = torch.tensor(self.actions, dtype=torch.long).to(self.device)
        rewards_t = torch.tensor(self.rewards, dtype=torch.float32).to(self.device)
        next_states_np = np.stack(self.next_states)  # shape: (batch, 1, 84, 84)
        next_states_t = torch.tensor(next_states_np, dtype=torch.float32).to(self.device) / 255.0
        dones_t = torch.tensor(self.dones, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            action_logits, state_values = self.network(states_t)
            _, next_state_values = self.network(next_states_t)
            dist = Categorical(logits=action_logits)
            log_probs_t = dist.log_prob(actions_t)
            advantages = self.compute_gae(
                rewards_t,
                state_values.squeeze(),
                next_state_values.squeeze(),
                dones_t
            )
            returns = advantages + state_values.squeeze()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = self.batch_size
        num_epochs = self.num_epochs
        indices = np.arange(len(states_t))

        for _ in range(num_epochs):
            np.random.shuffle(indices)
            for start_idx in range(0, len(states_t), batch_size):
                batch_idx = indices[start_idx:start_idx + batch_size]
                b_states = states_t[batch_idx]
                b_actions = actions_t[batch_idx]
                b_log_probs_old = log_probs_t[batch_idx]
                b_advantages = advantages[batch_idx]
                b_returns = returns[batch_idx]

                current_logits, current_values = self.network(b_states)
                dist_current = Categorical(logits=current_logits)
                current_log_probs = dist_current.log_prob(b_actions)

                ratios = torch.exp(current_log_probs - b_log_probs_old)
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * b_advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * (current_values.squeeze() - b_returns).pow(2).mean()

                entropy = dist_current.entropy().mean()
                total_loss = actor_loss + critic_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()

