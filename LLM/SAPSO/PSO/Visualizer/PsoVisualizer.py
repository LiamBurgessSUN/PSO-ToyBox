import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

from LLM.SAPSO.PSO.Visualizer import SwarmVisualizer


def plot_parameter_trajectory(history):
    omegas = [step["omega"] for step in history]
    c1s = [step["c1"] for step in history]
    c2s = [step["c2"] for step in history]

    plt.figure(figsize=(10, 4))
    plt.plot(omegas, label="ω (Inertia)", color='blue')
    plt.plot(c1s, label="c1 (Cognitive)", color='green')
    plt.plot(c2s, label="c2 (Social)", color='red')
    plt.xlabel("Step")
    plt.ylabel("Parameter Value")
    plt.title("SAC Agent Control Parameters Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def animate_swarm(env):
    if env.pso.dim != 2:
        print("Swarm visualization only works with 2D problems.")
        return

    visualizer = SwarmVisualizer(env.pso)

    def step_fn():
        if env.current_step < len(env.history):
            positions = env.history[env.current_step]["positions"]
            for i, p in enumerate(env.pso.particles):
                p.position = positions[i]
        env.current_step += 1

    visualizer.animate(step_fn, num_steps=len(env.history))

def smooth_with_std(data, window=50):
    smoothed = uniform_filter1d(data, size=window)
    std_dev = []
    half = window // 2
    for i in range(len(data)):
        start = max(0, i - half)
        end = min(len(data), i + half)
        std = np.std(data[start:end])
        std_dev.append(std)
    return smoothed, np.array(std_dev)

def plot_swarm_metrics(history, window=50):
    timesteps = np.arange(len(history))
    diversity = np.array([h["diversity"] for h in history])
    velocity = np.array([h["velocity"] for h in history])

    diversity_sm, diversity_std = smooth_with_std(diversity, window)
    velocity_sm, velocity_std = smooth_with_std(velocity, window)

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Diversity Plot
    axs[0].plot(timesteps, diversity, color='lightgray', alpha=0.5, label="Raw")
    axs[0].plot(timesteps, diversity_sm, label="Smoothed", color='tab:blue')
    axs[0].fill_between(timesteps,
                        diversity_sm - diversity_std,
                        diversity_sm + diversity_std,
                        alpha=0.3, color='tab:blue',
                        label="Std Dev")
    axs[0].set_ylabel("Diversity", fontsize=12)
    axs[0].legend()
    axs[0].grid(True)

    # Velocity Plot
    axs[1].plot(timesteps, velocity, color='lightgray', alpha=0.5, label="Raw")
    axs[1].plot(timesteps, velocity_sm, label="Smoothed", color='tab:green')
    axs[1].fill_between(timesteps,
                        velocity_sm - velocity_std,
                        velocity_sm + velocity_std,
                        alpha=0.3, color='tab:green',
                        label="Std Dev")
    axs[1].set_ylabel("Velocity", fontsize=12)
    axs[1].set_xlabel("Timestep", fontsize=12)
    axs[1].legend()
    axs[1].grid(True)

    # Axis formatting
    axs[1].set_xticks(np.linspace(0, len(timesteps), num=6, dtype=int))
    axs[0].set_yticks(np.round(np.linspace(min(diversity_sm), max(diversity_sm), num=5), 2))
    axs[1].set_yticks(np.round(np.linspace(min(velocity_sm), max(velocity_sm), num=5), 2))

    plt.suptitle("Swarm Metrics with Moving Average and Std Dev", fontsize=14)
    plt.tight_layout()
    plt.show()