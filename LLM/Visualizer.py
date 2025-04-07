import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

class SwarmVisualizer:
    def __init__(self, swarm, interval=100):
        self.swarm = swarm
        self.bounds = swarm.bounds
        self.interval = interval

        if swarm.dim != 2:
            raise ValueError("Visualizer only supports 2D swarms for now.")

        self.fig, self.ax = plt.subplots()
        self.particle_dots = None
        self.velocity_arrows = []
        self.gbest_dot = None

        self._setup_plot()

    def _setup_plot(self):
        self.ax.set_xlim(self.bounds[0], self.bounds[1])
        self.ax.set_ylim(self.bounds[0], self.bounds[1])
        self.ax.set_title("Swarm Optimization - Particle Movement")
        self.ax.set_xlabel("x1")
        self.ax.set_ylabel("x2")

        # Generate grid for contour plot
        x = np.linspace(self.bounds[0], self.bounds[1], 100)
        y = np.linspace(self.bounds[0], self.bounds[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.array([
            self.swarm.objective_function.evaluate(np.array([x_val, y_val]))
            for x_val, y_val in zip(np.ravel(X), np.ravel(Y))
        ]).reshape(X.shape)

        self.ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.6)
        self.ax.contour(X, Y, Z, levels=20, colors='k', linewidths=0.2, alpha=0.3)

        # Particle dots
        positions = np.array([p.position for p in self.swarm.particles])
        self.particle_dots, = self.ax.plot(positions[:, 0], positions[:, 1],
                                           'bo', label='Particles')

        # Velocity arrows
        for p in self.swarm.particles:
            arrow = self.ax.arrow(p.position[0], p.position[1],
                                  p.velocity[0], p.velocity[1],
                                  head_width=0.1, color='gray', alpha=0.5)
            self.velocity_arrows.append(arrow)

        # Global best
        self.gbest_dot, = self.ax.plot([], [], 'ro', label='Global Best')
        self.ax.legend()

    def _update_plot(self, _):
        # Update particle positions
        positions = np.array([p.position for p in self.swarm.particles])
        self.particle_dots.set_data(positions[:, 0], positions[:, 1])

        # Remove old arrows
        for arrow in self.velocity_arrows:
            arrow.remove()
        self.velocity_arrows.clear()

        # Draw new arrows
        for p in self.swarm.particles:
            arrow = self.ax.arrow(p.position[0], p.position[1],
                                  p.velocity[0], p.velocity[1],
                                  head_width=0.1, color='gray', alpha=0.5)
            self.velocity_arrows.append(arrow)

        # Update global best marker
        self.gbest_dot.set_data(
            [self.swarm.gbest_position[0]],
            [self.swarm.gbest_position[1]]
        )

    def animate(self, steps_fn, num_steps=100):
        """Run the animation using a function that performs swarm updates."""
        anim = animation.FuncAnimation(
            self.fig,
            lambda i: self._animate_step(steps_fn),
            frames=num_steps,
            interval=self.interval,
            blit=False  # You can set this to True with some refactoring
        )

        plt.show()

    def _animate_step(self, steps_fn):
        steps_fn()  # perform one step (e.g., swarm.optimize_step(...))
        self._update_plot(None)
