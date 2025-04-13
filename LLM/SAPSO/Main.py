import torch
import numpy as np
import matplotlib.pyplot as plt

from LLM.RL.ActorCritic.Agent import SACAgent
from LLM.RL.Replay.ReplayBuffer import ReplayBuffer
from LLM.SAPSO.PSOVisualizer import plot_parameter_trajectory, animate_swarm
from LLM.SAPSO.PSO_GYM import PSOEnv


def main():
    # === Setup ===
    env = PSOEnv(dim=30, num_particles=30, max_steps=100)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = SACAgent(state_dim, action_dim, device=device)
    buffer = ReplayBuffer(1_000_000, state_dim, action_dim, device=device)

    # === Training Hyperparameters ===
    num_episodes = 200
    batch_size = 256
    start_steps = 1000  # initial random exploration
    updates_per_step = 1

    rewards_per_episode = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0

        done = False
        while not done:
            if len(buffer) < start_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            next_state, reward, done, _, _ = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(buffer) >= batch_size:
                for _ in range(updates_per_step):
                    agent.update(buffer, batch_size)

        rewards_per_episode.append(total_reward)
        print(f"Episode {episode + 1} | Total Reward: {total_reward:.4f}")

    # === Save Model ===
    # agent.save("sac_psoenv_checkpoint.pth")

    # === Plot Reward Curve ===
    plt.plot(rewards_per_episode)
    plt.title("SAC on PSOEnv (Adaptive Control Params)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.show()

    plot_parameter_trajectory(env.history)
    animate_swarm(env)


if __name__ == "__main__":
    main()
