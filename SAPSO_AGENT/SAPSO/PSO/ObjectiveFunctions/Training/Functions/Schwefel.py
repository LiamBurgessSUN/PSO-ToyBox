import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class Schwefel1Function(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-100, 100)
        self.alpha = np.sqrt(np.pi)

    def evaluate(self, x: np.ndarray) -> float:
        return np.sum(x ** 2) ** self.alpha

    # --- Method for Schwefel1Function ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/Schwefel.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Schwefel N. 1 function. """
        # x_matrix shape: (num_particles, dim)
        sum_sq = np.sum(x_matrix ** 2, axis=1)  # Shape: (num_particles,)
        # Ensure non-negative base if alpha is not an integer
        sum_sq = np.maximum(sum_sq, 0)
        fitness_values = sum_sq ** self.alpha
        return fitness_values  # Shape: (num_particles,)