import gym
import torch
import matplotlib.pyplot as plt

from LLM.RL.ActorCritic.Agent import SACAgent
from LLM.RL.Replay.ReplayBuffer import ReplayBuffer

import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

def main():
    # === ENV & SEEDING ===
    env = gym.make("Pendulum-v1")
    seed = 42
    env.reset(seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # === ENV INFO ===
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # === AGENT & REPLAY BUFFER ===
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    agent = SACAgent(state_dim, action_dim, device="cuda")
    buffer = ReplayBuffer(capacity=1_000_000, state_dim=state_dim, action_dim=action_dim, device=device)

    # === HYPERPARAMETERS ===
    num_episodes = 100
    max_steps = 200
    batch_size = 256
    start_steps = 1000  # initial exploration
    updates_per_step = 1

    rewards = []

    # agent.load("sac_pendulum_checkpoint.pth")

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        # if episode % 5 == 0:
        #     env.render()

        for t in range(max_steps):
            # Explore initially, then use policy
            if len(buffer) < start_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if len(buffer) >= batch_size:
                for _ in range(updates_per_step):
                    agent.update(buffer, batch_size)

            if done:
                break

        rewards.append(episode_reward)
        print(f"Episode {episode + 1} | Reward: {episode_reward:.2f}")

    # === SAVE MODEL ===
    agent.save("sac_pendulum_checkpoint.pth")

    # === PLOT LEARNING CURVE ===
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("SAC on Pendulum-v1")
    plt.grid()
    plt.show()

    # agent.save("sac_pendulum_checkpoint.pth")


if __name__ == "__main__":
    main()
