#%%
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import gymnasium as gym
import pufferlib.emulation
from matplotlib import pyplot as plt

# Hyperparameters (tune these as needed for your project)
GAMMA = 0.99              # Discount factor
T_START = 10.0       # Initial exploration rate
T_END = 1.        # Final exploration rate
TEMPERATURE_DECAY = 0.995     # Exploration decay rate
BUFFER_SIZE = 100000      # Replay buffer size
BATCH_SIZE = 64           # Training batch size
LEARNING_RATE = 0.0005    # Optimizer learning rate
TARGET_UPDATE_FREQ = 100  # How often to update target network
MAX_EPISODES = 500        # Total training episodes
MAX_STEPS = 500           # Max steps per episode

device = "cuda"

# Define the Q-Network (PyTorch model)
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size+1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.clip(x, -4,8)
        mu, nu = torch.split(x, [x.shape[-1]-1,1], dim=-1)
        mu = torch.exp(mu)

        return mu, nu.squeeze(-1)


    def loss(self, r, mu_q, nu_q, mu_p, nu_p, dones, gamma):
        dones = dones.bool()
        z = (mu_q - r) * torch.exp(-nu_q)
        loss_done = nu_q - z + torch.exp(z.clip(max=12.))

        # with torch.no_grad():
        logsumexp = torch.logsumexp(mu_p * torch.exp(-nu_p).unsqueeze(1), dim=1)
        mu_p_target = r + gamma * torch.exp(nu_p) * logsumexp
        nu_p_target = nu_p + np.log(gamma)

        mu_p = mu_p_target
        nu_p = nu_p_target

        # Loss calculation based on the paper's formulas
        z = (mu_q - mu_p) * torch.exp(-nu_q)
        d = torch.exp(nu_p - nu_q)

        loss = nu_q - z + np.euler_gamma * d + torch.exp(z.clip(max=12.)) * torch.exp(torch.lgamma(1 + d))
        KL = loss - (nu_p + np.euler_gamma + 1)

        loss[dones] = loss_done[dones]
        KL[dones] = loss_done[dones]
        # print(f"loss: {loss.mean().item():.4f}, KL: {KL.mean().item():.4f}, KL_r!=0 : {KL[r!=0].mean().item():.4f}")

        return KL.mean(), loss

# Replay buffer for experience replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Function to create the environment (easily swappable)
def make_env(env_name='CartPole-v1'):
    gym_env = gym.make(env_name)
    env = pufferlib.emulation.GymnasiumPufferEnv(env_creator=lambda: gym_env)
    return env

# Main training function
def train_dqn(env_name='CartPole-v1'):
    env = make_env(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Initialize networks and optimizer
    policy_net = QNetwork(state_size, action_size).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    temperature = T_START
    step_count = 0
    last_loss, last_KL = 0, 0
    episode_rewards=[]

    for episode in range(MAX_EPISODES):
        state, _ = env.reset()
        state = np.array(state)  # Ensure numpy array
        episode_reward = 0
        done = False

        for step in range(MAX_STEPS):
            step_count += 1

            # Epsilon-greedy action selection
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(device)
                mu, nu = policy_net(state_tensor)
                beta = torch.exp(nu).unsqueeze(-1)
                logits = (mu / beta) / temperature
                action_dist = torch.distributions.Categorical(logits=logits)
                action = action_dist.sample().cpu().item()

            # Step the environment
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.array(next_state)
            episode_reward += reward

            # Push to replay buffer
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state

            # Train if buffer is large enough
            if len(replay_buffer) > BATCH_SIZE:
                batch = replay_buffer.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(np.array(states)).squeeze(1).to(device)
                actions = torch.LongTensor([*actions]).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(np.array(next_states)).squeeze(1).to(device)
                dones = torch.FloatTensor(dones).to(device)

                # Compute Q values
                mu_q, nu_q = policy_net(states)
                mu_p, nu_p = policy_net(next_states)
                mu_q = mu_q[torch.arange(BATCH_SIZE), actions]
                
                KL_loss, Entropy_Loss = policy_net.loss(rewards, mu_q, nu_q, mu_p, nu_p, dones, GAMMA)
                # Loss and optimization
                optimizer.zero_grad()
                KL_loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 1.)
                optimizer.step()
                last_loss = KL_loss.item()
                last_KL = Entropy_Loss.mean().item()

            if done:
                break

        # Decay epsilon
        temperature = max(T_END, temperature * TEMPERATURE_DECAY)
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{MAX_EPISODES} | Reward: {episode_reward:.1f} | Temperature: {temperature:.3f} | KL_loss: {last_loss:.3f}, Entropy: {last_KL:.3f}")

    env.close()
    return policy_net, episode_rewards  # Return trained model if needed

# Run the training
if __name__ == "__main__":
    trained_model, rewards = train_dqn(env_name='CartPole-v1')  # Swap to e.g., 'MountainCar-v0' for another env
    with open('q-prob_rewards.npy', 'wb') as f:
        np.save(f, rewards)