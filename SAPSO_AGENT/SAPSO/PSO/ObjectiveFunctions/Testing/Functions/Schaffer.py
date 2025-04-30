import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class Schaffer4Function(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-100, 100)

    def evaluate(self, x: np.ndarray) -> float:
        return np.sum([
            0.5 + (np.cos(np.sin(x[j] ** 2 - x[j + 1] ** 2)) ** 2 - 0.5) /
            (1 + 0.001 * (x[j] ** 2 + x[j + 1] ** 2)) ** 2
            for j in range(len(x) - 1)
        ])

    # --- Method for Schaffer4Function ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Testing/Schaffer.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Schaffer N. 4 function. """
        # x_matrix shape: (num_particles, dim)
        x_j = x_matrix[:, :-1]  # Shape: (num_particles, dim-1)
        x_j_plus_1 = x_matrix[:, 1:]  # Shape: (num_particles, dim-1)

        term1 = x_j ** 2
        term2 = x_j_plus_1 ** 2

        cos_term = np.cos(np.sin(term1 - term2)) ** 2
        denominator = (1 + 0.001 * (term1 + term2)) ** 2

        # Calculate sum across the dimension pairs for each particle
        fitness_values = np.sum(0.5 + (cos_term - 0.5) / denominator, axis=1)  # Shape: (num_particles,)
        return fitness_values