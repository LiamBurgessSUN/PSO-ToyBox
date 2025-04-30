import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class BohachevskyFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-15, 15)  # Correct bounds from the definition

    def evaluate(self, x: np.ndarray) -> float:
        total = 0.0
        for j in range(len(x) - 1):
            xj = x[j]
            xjp1 = x[j + 1]
            term = (
                xj ** 2 +
                2 * xjp1 ** 2 -
                0.3 * np.cos(3 * np.pi * xj) -
                0.4 * np.cos(4 * np.pi * xjp1) +
                0.7
            )
            total += term
        return total

    # --- Method for BohachevskyFunction ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/Bohachevsky.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Bohachevsky function. """
        # x_matrix shape: (num_particles, dim)
        x_j = x_matrix[:, :-1]  # Shape: (num_particles, dim-1)
        x_j_plus_1 = x_matrix[:, 1:]  # Shape: (num_particles, dim-1)

        term = (x_j ** 2 +
                2 * x_j_plus_1 ** 2 -
                0.3 * np.cos(3 * np.pi * x_j) -
                0.4 * np.cos(4 * np.pi * x_j_plus_1) +
                0.7)  # Shape: (num_particles, dim-1)

        fitness_values = np.sum(term, axis=1)  # Shape: (num_particles,)
        return fitness_values
