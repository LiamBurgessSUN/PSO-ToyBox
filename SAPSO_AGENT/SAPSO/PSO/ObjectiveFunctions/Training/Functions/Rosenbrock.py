import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class RosenbrockFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-30, 30)

    def evaluate(self, x: np.ndarray) -> float:
        return np.sum([100 * (x[j + 1] - x[j] ** 2) ** 2 + (x[j] - 1) ** 2 for j in range(len(x) - 1)])

    # --- Method for RosenbrockFunction ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/Rosenbrock.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Rosenbrock function. """
        # x_matrix shape: (num_particles, dim)
        x_j = x_matrix[:, :-1]  # Shape: (num_particles, dim-1)
        x_j_plus_1 = x_matrix[:, 1:]  # Shape: (num_particles, dim-1)

        term1 = 100.0 * (x_j_plus_1 - x_j ** 2) ** 2
        term2 = (x_j - 1.0) ** 2

        # Sum across dimension pairs for each particle
        fitness_values = np.sum(term1 + term2, axis=1)  # Shape: (num_particles,)
        return fitness_values