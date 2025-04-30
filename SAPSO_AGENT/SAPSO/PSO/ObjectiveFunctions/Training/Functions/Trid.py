import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class TridFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-20, 20)

    def evaluate(self, x: np.ndarray) -> float:
        return np.sum((x - 1) ** 2) - np.sum(x[1:] * x[:-1])

    # --- Method for TridFunction ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/Trid.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Trid function. """
        # x_matrix shape: (num_particles, dim)
        term1_sum = np.sum((x_matrix - 1.0) ** 2, axis=1)  # Shape: (num_particles,)

        # Term 2: Sum of x[j] * x[j-1] for j=1 to dim-1
        x_j = x_matrix[:, 1:]  # x[1] to x[dim-1] -> Shape: (num_particles, dim-1)
        x_j_minus_1 = x_matrix[:, :-1]  # x[0] to x[dim-2] -> Shape: (num_particles, dim-1)
        term2_sum = np.sum(x_j * x_j_minus_1, axis=1)  # Shape: (num_particles,)

        fitness_values = term1_sum - term2_sum
        return fitness_values  # Shape: (num_particles,)