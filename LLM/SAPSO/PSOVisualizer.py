import matplotlib.pyplot as plt
import numpy as np


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
