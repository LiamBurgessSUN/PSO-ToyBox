import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class Schubert4Function(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-10, 10)

    def evaluate(self, x: np.ndarray) -> float:
        return np.sum([
            np.sum([(j + 1) * np.cos((j + 1) * xj + j) for j in range(5)])
            for xj in x
        ])

    # --- Method for Schubert4Function ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/Schubert.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Schubert function. """
        # x_matrix shape: (num_particles, dim)
        num_particles, dim = x_matrix.shape

        # Inner sum: sum_{j=1}^{5} (j+1) * cos((j+1)*x_k + j) for each x_k
        j_vals = np.arange(1, 6)  # j runs from 0 to 4 in the code (range(5)), formula uses j=1 to 5?
        # Let's assume code's range(5) means indices 0,1,2,3,4 for j.
        # Formula uses j+1, so (j+1) -> 1,2,3,4,5
        # Formula uses cos((j+1)*xk + j), so j -> 0,1,2,3,4
        j_inner = np.arange(5)  # Indices 0, 1, 2, 3, 4
        j_plus_1 = j_inner + 1  # Values 1, 2, 3, 4, 5

        # Reshape for broadcasting:
        # x_matrix: (num_particles, dim, 1)
        # j_plus_1: (1, 1, 5)
        # j_inner: (1, 1, 5)
        cos_term_arg = j_plus_1 * x_matrix[:, :, np.newaxis] + j_inner  # Shape: (num_particles, dim, 5)
        inner_sum_terms = j_plus_1 * np.cos(cos_term_arg)  # Shape: (num_particles, dim, 5)

        # Sum the inner part (over j=0 to 4)
        sum_over_j = np.sum(inner_sum_terms, axis=2)  # Shape: (num_particles, dim)

        # Sum the outer part (over dimensions k=0 to dim-1)
        fitness_values = np.sum(sum_over_j, axis=1)  # Shape: (num_particles,)
        return fitness_values