import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class LevyMontalvo2Function(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-5, 5)

    def evaluate(self, x: np.ndarray) -> float:
        term1 = 0.1 * np.sin(3 * np.pi * x[0]) ** 2
        term2 = np.sum((x[:-1] - 1) ** 2 * (np.sin(3 * np.pi * x[1:]) ** 2 + 1))
        term3 = (x[-1] - 1) ** 2 * (np.sin(2 * np.pi * x[-1]) ** 2 + 1)
        return term1 + 0.1 * term2 + 0.1 * term3

    # --- Method for LevyMontalvo2Function ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/LevyMontalvo.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Levy Montalvo N. 2 function. """
        # x_matrix shape: (num_particles, dim)
        x_1 = x_matrix[:, 0]  # First element, shape: (num_particles,)
        x_j = x_matrix[:, :-1]  # Shape: (num_particles, dim-1)
        x_j_plus_1 = x_matrix[:, 1:]  # Shape: (num_particles, dim-1)
        x_n = x_matrix[:, -1]  # Last element, shape: (num_particles,)

        term1 = 0.1 * np.sin(3.0 * np.pi * x_1) ** 2  # Shape: (num_particles,)
        term2_sum = 0.1 * np.sum((x_j - 1.0) ** 2 * (np.sin(3.0 * np.pi * x_j_plus_1) ** 2 + 1.0),
                                 axis=1)  # Shape: (num_particles,)
        term3 = 0.1 * (x_n - 1.0) ** 2 * (np.sin(2.0 * np.pi * x_n) ** 2 + 1.0)  # Shape: (num_particles,)

        fitness_values = term1 + term2_sum + term3
        return fitness_values  # Shape: (num_particles,)