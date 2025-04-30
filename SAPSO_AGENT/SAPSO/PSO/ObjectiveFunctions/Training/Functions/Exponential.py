import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class ExponentialFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-1, 1)

    def evaluate(self, x: np.ndarray) -> float:
        return -np.exp(-0.5 * np.sum(x ** 2))

    # --- Method for ExponentialFunction ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/Exponential.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Exponential function. """
        # x_matrix shape: (num_particles, dim)
        sum_sq = np.sum(x_matrix ** 2, axis=1)  # Shape: (num_particles,)
        fitness_values = -np.exp(-0.5 * sum_sq)
        return fitness_values