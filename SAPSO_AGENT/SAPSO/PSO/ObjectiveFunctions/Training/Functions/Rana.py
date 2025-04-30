import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class RanaFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-500, 500)

    def evaluate(self, x: np.ndarray) -> float:
        sum_ = 0
        for j in range(len(x) - 1):
            xj = x[j]
            xj1 = x[j + 1]
            t1 = np.sqrt(np.abs(xj1 + xj + 1))
            t2 = np.sqrt(np.abs(xj1 - xj + 1))
            sum_ += (xj1 + 1) * np.cos(t2) * np.sin(t1) + xj * np.cos(t1) * np.sin(t2)
        return sum_

    # --- Method for RanaFunction ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/Rana.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Rana function. """
        # x_matrix shape: (num_particles, dim)
        x_j = x_matrix[:, :-1]  # Shape: (num_particles, dim-1)
        x_j_plus_1 = x_matrix[:, 1:]  # Shape: (num_particles, dim-1)

        # Calculate t1 and t2, ensuring non-negative args for sqrt
        t1_arg = np.abs(x_j_plus_1 + x_j + 1.0)
        t2_arg = np.abs(x_j_plus_1 - x_j + 1.0)
        t1 = np.sqrt(t1_arg)
        t2 = np.sqrt(t2_arg)

        # Calculate terms in the sum
        term = (x_j_plus_1 + 1.0) * np.cos(t2) * np.sin(t1) + x_j * np.cos(t1) * np.sin(
            t2)  # Shape: (num_particles, dim-1)

        # Sum across dimension pairs for each particle
        fitness_values = np.sum(term, axis=1)  # Shape: (num_particles,)
        return fitness_values