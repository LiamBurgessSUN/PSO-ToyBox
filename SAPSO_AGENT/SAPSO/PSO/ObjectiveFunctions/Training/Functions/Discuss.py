import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class DiscussFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-100, 100)

    def evaluate(self, x: np.ndarray) -> float:
        return 1e6 * x[0] ** 2 + np.sum(x[1:] ** 2)

    # --- Method for DiscussFunction ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/Discuss.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Discuss function. """
        # x_matrix shape: (num_particles, dim)
        term1 = 1e6 * x_matrix[:, 0] ** 2  # Shape: (num_particles,) - First dimension only
        # Sum remaining dimensions squared for each particle
        term2 = np.sum(x_matrix[:, 1:] ** 2, axis=1)  # Shape: (num_particles,)
        fitness_values = term1 + term2
        return fitness_values