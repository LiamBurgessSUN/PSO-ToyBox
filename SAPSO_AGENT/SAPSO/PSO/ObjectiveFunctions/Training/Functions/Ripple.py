import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class Ripple25Function(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (0, 1)

    def evaluate(self, x: np.ndarray) -> float:
        return np.sum([
            -np.exp(-2 * np.log(2) * ((xj - 0.1) / 0.8) ** 2) * (np.sin(5 * np.pi * xj) ** 6)
            for xj in x
        ])

    # --- Method for Ripple25Function ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/Ripple.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Ripple N. 25 function. """
        # x_matrix shape: (num_particles, dim)
        term_in_pow = (x_matrix - 0.1) / 0.8
        # Avoid issues with log2(0) if x_j happens to be 0.1 (unlikely with floats, but safe)
        term_in_pow[term_in_pow == 0] = np.finfo(float).eps

        exp_arg = -2.0 * np.log2(np.maximum(np.abs(term_in_pow), np.finfo(
            float).eps)) * 2  # Original formula had log(2)*()^2, seems like log2 was intended? Using log2. Original: np.log(2) * term_in_pow**2
        # Using formula as appears in file: -2 * log(2) * ((xj - 0.1) / 0.8)**2
        exp_arg_original = -2.0 * np.log(2) * term_in_pow ** 2

        sin_term = np.sin(5 * np.pi * x_matrix) ** 6

        # Calculate sum term for each particle
        term = -np.exp(exp_arg_original) * sin_term  # Shape: (num_particles, dim)
        fitness_values = np.sum(term, axis=1)  # Shape: (num_particles,)
        return fitness_values