import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class WeierstrassFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30, a=0.5, b=3, j_max=20):
        super().__init__(dim, num_particles)
        self.a = a
        self.b = b
        self.j_max = j_max
        self.bounds = (-0.5, 0.5)

    def evaluate(self, x: np.ndarray) -> float:
        n = len(x)
        term1 = np.sum([
            np.sum([self.a ** j * np.cos(2 * np.pi * self.b ** j * (xj + 0.5)) for j in range(self.j_max + 1)])
            for xj in x
        ])
        term2 = n * np.sum([self.a ** j * np.cos(np.pi * self.b ** j) for j in range(self.j_max + 1)])
        return term1 - term2

    # --- Method for WeierstrassFunction ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/Weierstrass.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Weierstrass function. """
        # x_matrix shape: (num_particles, dim)
        num_particles, dim = x_matrix.shape
        if dim != self.dim:
            pass  # Handle dim mismatch
        n = self.dim

        # Precompute powers of a and b for the inner sum
        j_vals = np.arange(self.j_max + 1)  # Indices 0, 1, ..., j_max
        a_pow_j = self.a ** j_vals  # Shape: (j_max+1,)
        b_pow_j = self.b ** j_vals  # Shape: (j_max+1,)

        # Calculate the first main term (sum over dimensions k, inner sum over j)
        # Reshape for broadcasting:
        # x_matrix: (num_particles, dim, 1)
        # a_pow_j: (1, 1, j_max+1)
        # b_pow_j: (1, 1, j_max+1)
        cos_arg = 2 * np.pi * b_pow_j * (x_matrix[:, :, np.newaxis] + 0.5)  # Shape: (num_particles, dim, j_max+1)
        inner_sum_term = a_pow_j * np.cos(cos_arg)  # Shape: (num_particles, dim, j_max+1)

        # Sum inner part (over j)
        sum_over_j = np.sum(inner_sum_term, axis=2)  # Shape: (num_particles, dim)
        # Sum outer part (over dimensions k)
        term1 = np.sum(sum_over_j, axis=1)  # Shape: (num_particles,)

        # Calculate the second main term (constant offset for all particles)
        term2_inner_sum = np.sum(a_pow_j * np.cos(np.pi * b_pow_j))  # Scalar value
        term2 = n * term2_inner_sum  # Scalar value

        fitness_values = term1 - term2
        return fitness_values  # Shape: (num_particles,)