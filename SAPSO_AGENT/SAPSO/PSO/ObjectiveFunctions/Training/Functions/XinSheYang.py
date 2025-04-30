import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class XinSheYang1Function(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-5, 5)
        self.epsilon = np.random.uniform(0, 1, dim)

    def evaluate(self, x: np.ndarray) -> float:
        return np.sum([self.epsilon[j] * np.abs(x[j]) ** (j + 1) for j in range(len(x))])

    # --- Method for XinSheYang1Function ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/XinSheYang.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Xin-She Yang N. 1 function. """
        # x_matrix shape: (num_particles, dim)
        num_particles, dim = x_matrix.shape
        if dim != self.dim:
            pass  # Handle dim mismatch

        # Ensure epsilon has the correct dimension
        if not hasattr(self, 'epsilon') or len(self.epsilon) != self.dim:
            # Initialize epsilon if not present or dimension mismatch
            # print("Warning: Re-initializing epsilon for XinSheYang1Function") # Optional debug
            self.epsilon = np.random.uniform(0, 1, self.dim)

        j_plus_1 = np.arange(1, self.dim + 1)  # Exponents 1, 2, ..., dim

        # Calculate term inside sum: epsilon_j * |x_j|^(j+1)
        # Reshape for broadcasting:
        # self.epsilon: (dim,)
        # np.abs(x_matrix): (num_particles, dim)
        # j_plus_1: (dim,)
        term = self.epsilon * np.abs(x_matrix) ** j_plus_1  # Shape: (num_particles, dim)

        # Sum across dimensions for each particle
        fitness_values = np.sum(term, axis=1)  # Shape: (num_particles,)
        return fitness_values


class XinSheYang2Function(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-2 * np.pi, 2 * np.pi)

    def evaluate(self, x: np.ndarray) -> float:
        abs_sum = np.sum(np.abs(x))
        sin_sq_sum = np.sum(np.sin(x ** 2))
        return abs_sum * np.exp(-sin_sq_sum)

    # --- Method for XinSheYang2Function ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/XinSheYang.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Xin-She Yang N. 2 function. """
        # x_matrix shape: (num_particles, dim)
        sum_abs_x = np.sum(np.abs(x_matrix), axis=1)  # Shape: (num_particles,)
        sum_sin_sq = np.sum(np.sin(x_matrix ** 2), axis=1)  # Shape: (num_particles,)

        fitness_values = sum_abs_x * np.exp(-sum_sin_sq)
        return fitness_values  # Shape: (num_particles,)

