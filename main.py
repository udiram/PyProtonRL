import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from gym import spaces
import matplotlib.pyplot as plt
from copy import deepcopy


# Define the Treatment Planning Environment
class TreatmentPlanningEnv(gym.Env):
    """Environment for proton PBS treatment planning optimization."""

    def __init__(self, num_oars=5):
        super(TreatmentPlanningEnv, self).__init__()
        self.prescription_dose = 50.0  # Gy
        self.num_oars = num_oars
        self.max_steps = 10
        self.current_step = 0

        # Initial weights for OARs (simulating Plan_LBFGS)
        self.weights_oar = np.ones(num_oars) * 0.5

        # Action space: change in OAR weights [-1, 1] per OAR
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(num_oars,), dtype=np.float32)

        # State space: [D98, D50, D2 for CTV, mean_dose for each OAR]
        state_dim = 3 + num_oars
        self.observation_space = spaces.Box(low=0.0, high=100.0, shape=(state_dim,), dtype=np.float32)

        self.state = self._compute_state()

    def _compute_state(self):
        """Simulate dose distribution based on OAR weights."""
        # CTV doses influenced by total OAR weighting
        total_oar_weight = np.sum(self.weights_oar)
        D98_ctv = max(50.0 - 1.0 * total_oar_weight, 0.0)
        D50_ctv = 50.0  # Fixed median dose
        D2_ctv = min(50.0 + 1.0 * total_oar_weight, 100.0)

        # OAR mean doses decrease with individual weights
        mean_doses_oar = [max(30.0 - 10.0 * w, 0.0) for w in self.weights_oar]

        return np.array([D98_ctv, D50_ctv, D2_ctv] + mean_doses_oar)

    def reset(self):
        """Reset to initial Plan_LBFGS-like state."""
        self.weights_oar = np.ones(self.num_oars) * 0.5
        self.current_step = 0
        self.state = self._compute_state()
        return self.state

    def step(self, action):
        """Apply action, update state, and compute reward."""
        self.weights_oar = np.clip(self.weights_oar + action, 0.0, 10.0)
        self.state = self._compute_state()
        self.current_step += 1

        reward = self._calculate_reward()
        done = self.current_step >= self.max_steps or reward == 0  # Stop if perfect or max steps reached

        return self.state, reward, done, {}

    def _calculate_reward(self):
        """Compute reward based on dose distribution (Figure 4 in paper)."""
        D98_ctv = self.state[0]
        mean_doses_oar = self.state[3:]

        # Clinical goals
        ctv_min = 0.95 * self.prescription_dose  # D98 >= 47.5 Gy
        oar_max = 20.0  # Mean dose <= 20 Gy for each OAR

        # "No penalty" interval (simplified from paper's intent)
        ctv_penalty = max(0, ctv_min - D98_ctv)
        oar_penalties = [max(0, dose - oar_max) for dose in mean_doses_oar]

        # Quadratic penalty
        total_penalty = ctv_penalty ** 2 + sum(p ** 2 for p in oar_penalties)
        return -total_penalty


# Transformer-Based Actor-Critic Model
class TransformerActorCritic(nn.Module):
    """Transformer-based actor-critic agent (Figure 3 in paper)."""

    def __init__(self, state_dim, action_dim, nhead=2, num_layers=2, hidden_dim=64):
        super(TransformerActorCritic, self).__init__()

        # Input embedding
        self.embedding = nn.Linear(state_dim, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Actor head: outputs mean and log_std for action distribution
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head: outputs state value
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        """Forward pass for actor and critic."""
        # State shape: (batch_size, state_dim)
        x = self.embedding(state)
        x = self.transformer(x.unsqueeze(1)).squeeze(1)  # Treat state as single sequence

        # Actor
        mean = self.actor_mean(x)
        std = torch.exp(self.actor_log_std)
        dist = Normal(mean, std)

        # Critic
        value = self.critic(x)

        return dist, value


# PPO Implementation
class PPO:
    """Proximal Policy Optimization trainer (Figure 6 in paper)."""

    def __init__(self, env, state_dim, action_dim, lr=3e-4, clip_eps=0.2, gamma=0.99, gae_lambda=0.95):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TransformerActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def collect_trajectory(self, max_steps=1000):
        """Collect a single trajectory."""
        states, actions, rewards, values, dones = [], [], [], [], []
        state = self.env.reset()

        for _ in range(max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            dist, value = self.model(state_tensor)
            action = dist.sample().cpu().numpy()[0]
            next_state, reward, done, _ = self.env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value.item())
            dones.append(done)

            state = next_state
            if done:
                break

        return states, actions, rewards, values, dones

    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        next_value = 0 if dones[-1] else self.model(torch.FloatTensor(self.env.state).unsqueeze(0).to(self.device))[
            1].item()

        for r, v, d in reversed(list(zip(rewards, values, dones))):
            delta = r + self.gamma * next_value * (1 - d) - v
            gae = delta + self.gamma * self.gae_lambda * (1 - d) * gae
            advantages.insert(0, gae)
            next_value = v

        returns = [a + v for a, v in zip(advantages, values)]
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        return advantages, returns

    def update(self, states, actions, advantages, returns, epochs=10, batch_size=32):
        """Update policy and value networks."""
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(epochs):
            for idx in range(0, len(states), batch_size):
                batch_states = states[idx:idx + batch_size]
                batch_actions = actions[idx:idx + batch_size]
                batch_advantages = advantages[idx:idx + batch_size]
                batch_returns = returns[idx:idx + batch_size]

                dist, values = self.model(batch_states)
                log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                old_dist = Normal(dist.mean.detach(), dist.stddev.detach())
                old_log_probs = old_dist.log_prob(batch_actions).sum(dim=-1)

                # PPO objective
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = (batch_returns - values.squeeze()).pow(2).mean()

                # Total loss
                loss = actor_loss + 0.5 * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def train(self, episodes=100):
        """Train the PPO agent."""
        for episode in range(episodes):
            states, actions, rewards, values, dones = self.collect_trajectory()
            advantages, returns = self.compute_gae(rewards, values, dones)
            self.update(states, actions, advantages, returns)

            avg_reward = np.mean(rewards)
            print(f"Episode {episode + 1}/{episodes}, Average Reward: {avg_reward:.2f}")


# Visualization Function
def plot_dvh(states_history):
    """Plot DVH-like metrics over optimization steps."""
    steps = range(len(states_history))
    ctv_d98 = [s[0] for s in states_history]
    oar_means = np.array([s[3:] for s in states_history]).T

    plt.figure(figsize=(12, 6))
    plt.plot(steps, ctv_d98, label="CTV D98", color="blue")
    for i, oar_mean in enumerate(oar_means):
        plt.plot(steps, oar_mean, label=f"OAR {i + 1} Mean Dose", linestyle="--")

    plt.axhline(y=47.5, color="red", linestyle="--", label="CTV D98 Min (47.5 Gy)")
    plt.axhline(y=20.0, color="green", linestyle="--", label="OAR Mean Max (20 Gy)")
    plt.xlabel("Optimization Step")
    plt.ylabel("Dose (Gy)")
    plt.title("Dose Metrics During Optimization")
    plt.legend()
    plt.grid(True)
    plt.show()


# Main Execution
def main():
    num_oars = 5  # Example with 5 OARs (e.g., Spinal Cord, Larynx, etc.)
    env = TreatmentPlanningEnv(num_oars=num_oars)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    ppo = PPO(env, state_dim, action_dim)

    # Collect initial trajectory for demonstration
    states_history = [env.reset()]
    state = env.reset()
    done = False
    device = ppo.device
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = ppo.model(state_tensor)
        action = dist.mean.detach().cpu().numpy()[0]  # Fixed with .detach()
        next_state, reward, done, _ = env.step(action)
        states_history.append(deepcopy(state))

    # Train the agent
    ppo.train(episodes=50)

    # Visualize results
    plot_dvh(states_history)


if __name__ == "__main__":
    main()