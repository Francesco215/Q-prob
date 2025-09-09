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

# Hyperparameters (tune these as needed for your project)
GAMMA = 0.99              # Discount factor
EPSILON_START = 1.0       # Initial exploration rate
EPSILON_END = 0.01        # Final exploration rate
EPSILON_DECAY = 0.995     # Exploration decay rate
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
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

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
    target_net = QNetwork(state_size, action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    epsilon = EPSILON_START
    step_count = 0
    loss = torch.tensor(0)
    episode_rewards = []

    for episode in range(MAX_EPISODES):
        state, _ = env.reset()
        state = np.array(state)  # Ensure numpy array
        episode_reward = 0
        done = False

        for step in range(MAX_STEPS):
            step_count += 1

            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, action_size - 1)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = policy_net(state_tensor).argmax().item()

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
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(np.array(next_states)).squeeze(1).to(device)
                dones = torch.FloatTensor(dones).to(device)

                # Compute Q values
                current_q = policy_net(states)[torch.arange(BATCH_SIZE), actions]
                with torch.no_grad():
                    next_q = target_net(next_states).max(1)[0]
                    target_q = rewards + (1 - dones) * GAMMA * next_q

                # Loss and optimization
                loss = F.mse_loss(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update target network periodically
            if step_count % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{MAX_EPISODES} | Reward: {episode_reward} | Epsilon: {epsilon:.3f} | Loss: {loss.item():.3f}")

    env.close()
    return policy_net, episode_rewards  # Return trained model if needed

# Run the training
if __name__ == "__main__":
    trained_model, rewards = train_dqn(env_name='CartPole-v1')  # Swap to e.g., 'MountainCar-v0' for another env
    with open('q-prob_rewards_base.npy', 'wb') as f:
        np.save(f, rewards)
# %%