import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class SalomonFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-100, 100)

    def evaluate(self, x: np.ndarray) -> float:
        sum_sq = np.sum(x ** 2)
        return -np.cos(2 * np.pi * np.sqrt(sum_sq)) + 0.1 * np.sqrt(sum_sq) + 1

    # --- Method for SalomonFunction ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/Salomon.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Salomon function. """
        # x_matrix shape: (num_particles, dim)
        sum_sq = np.sum(x_matrix ** 2, axis=1)  # Shape: (num_particles,)
        # Ensure non-negative for sqrt
        sum_sq_sqrt = np.sqrt(np.maximum(sum_sq, 0))  # Shape: (num_particles,)

        term1 = -np.cos(2 * np.pi * sum_sq_sqrt)
        term2 = 0.1 * sum_sq_sqrt
        term3 = 1.0

        # Note: Original formula often cited as f(x) = 1 - cos(...) + 0.1 * sqrt(...)
        # The provided code implements -cos(...) + 0.1 * sqrt(...) + 1
        # Sticking to the formula in the provided file:
        fitness_values = term1 + term2 + term3
        return fitness_values  # Shape: (num_particles,)