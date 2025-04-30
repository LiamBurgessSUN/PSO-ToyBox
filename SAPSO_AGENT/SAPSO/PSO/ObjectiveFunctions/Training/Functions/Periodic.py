import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class PeriodicFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-10, 10)

    def evaluate(self, x: np.ndarray) -> float:
        sum_sq = np.sum(x ** 2)
        return 1 + np.sum(np.sin(x) ** 2) - 0.1 * np.exp(-sum_sq)

    # --- Method for PeriodicFunction ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/Periodic.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Periodic function. """
        # x_matrix shape: (num_particles, dim)
        sum_sq = np.sum(x_matrix ** 2, axis=1)  # Shape: (num_particles,)
        sum_sin_sq = np.sum(np.sin(x_matrix) ** 2, axis=1)  # Shape: (num_particles,)

        fitness_values = 1.0 + sum_sin_sq - 0.1 * np.exp(-sum_sq)
        return fitness_values  # Shape: (num_particles,)