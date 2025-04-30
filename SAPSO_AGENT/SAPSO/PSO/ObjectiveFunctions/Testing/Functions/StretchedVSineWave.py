import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class StretchedVSineWaveFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-10, 10)

    def evaluate(self, x: np.ndarray) -> float:
        return np.sum([
            (x[j] ** 2 + x[j + 1] ** 2) ** 0.25 *
            (np.sin(50 * (x[j] ** 2 + x[j + 1] ** 2) ** 0.1) ** 2 + 0.1)
            for j in range(len(x) - 1)
        ])

    # --- Method for StretchedVSineWaveFunction ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Testing/StretchedVSineWave.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Stretched V Sine Wave function. """
        # x_matrix shape: (num_particles, dim)
        x_j = x_matrix[:, :-1]  # Shape: (num_particles, dim-1)
        x_j_plus_1 = x_matrix[:, 1:]  # Shape: (num_particles, dim-1)

        term_sq_sum = x_j ** 2 + x_j_plus_1 ** 2
        term_pow_01 = term_sq_sum ** 0.1
        term_pow_025 = term_sq_sum ** 0.25

        sin_term = np.sin(50 * term_pow_01) ** 2

        # Calculate sum across the dimension pairs for each particle
        fitness_values = np.sum(term_pow_025 * (sin_term + 0.1), axis=1)  # Shape: (num_particles,)
        return fitness_values