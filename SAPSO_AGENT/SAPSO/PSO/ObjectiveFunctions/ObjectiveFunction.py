# --- Objective Function Base Class ---
from abc import ABC, abstractmethod

import numpy as np
from matplotlib import pyplot as plt


class ObjectiveFunction(ABC):
    def __init__(self, dim=30, num_particles=30):
        self.dim = dim
        self.num_particles = num_particles
        self.bounds = (-5.12, 5.12)  # Default bounds

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> float:
        pass

    def plot_3d_surface(self, resolution=100):
        if self.dim != 2:
            raise ValueError("3D surface plot only supports 2D objective functions.")

        x = np.linspace(self.bounds[0], self.bounds[1], resolution)
        y = np.linspace(self.bounds[0], self.bounds[1], resolution)
        X, Y = np.meshgrid(x, y)

        Z = np.array([
            self.evaluate(np.array([x_val, y_val]))
            for x_val, y_val in zip(np.ravel(X), np.ravel(Y))
        ]).reshape(X.shape)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', alpha=0.8)
        ax.set_title("3D Surface of Objective Function")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("f(x)")

        plt.show()

    @abstractmethod
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """
        Evaluates the objective function for multiple particle positions simultaneously.

        Args:
            x_matrix (np.ndarray): A 2D NumPy array where each row represents a
                                   particle's position vector.
                                   Shape: (num_particles, dim).

        Returns:
            np.ndarray: A 1D NumPy array containing the fitness value for each
                        particle position in the input matrix.
                        Shape: (num_particles,).
        """
        # Default implementation raises NotImplementedError if a subclass
        # doesn't override it but is called.
        # Alternatively, subclasses can implement this, or you could provide
        # a default implementation that loops through evaluate() if desired,
        # though that defeats the purpose of vectorization.
        raise NotImplementedError("Subclasses must implement the vectorized evaluate_matrix method.")

