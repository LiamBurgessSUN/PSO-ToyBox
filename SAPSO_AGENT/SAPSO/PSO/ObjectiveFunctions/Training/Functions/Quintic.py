import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class QuinticFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-10, 10)

    def evaluate(self, x: np.ndarray) -> float:
        return np.sum(np.abs(x ** 5 - 3 * x ** 4 + 4 * x ** 3 + 2 * x ** 2 - 10 * x - 4))

    # --- Method for QuinticFunction ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/Quintic.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Quintic function. """
        # x_matrix shape: (num_particles, dim)
        term = np.abs(
            x_matrix ** 5 - 3.0 * x_matrix ** 4 + 4.0 * x_matrix ** 3 + 2.0 * x_matrix ** 2 - 10.0 * x_matrix - 4.0)
        # Sum across dimensions for each particle
        fitness_values = np.sum(term, axis=1)  # Shape: (num_particles,)
        return fitness_values