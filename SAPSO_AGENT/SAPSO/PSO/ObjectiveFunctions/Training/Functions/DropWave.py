import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class DropWaveFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-5.12, 5.12)

    def evaluate(self, x: np.ndarray) -> float:
        sum_sq = np.sum(x ** 2)
        numerator = 1 + np.cos(12 * np.sqrt(sum_sq))
        denominator = 2 + 0.5 * sum_sq
        return -numerator / denominator

    # --- Method for DropWaveFunction ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/DropWave.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Drop-Wave function. """
        # x_matrix shape: (num_particles, dim)
        sum_sq = np.sum(x_matrix ** 2, axis=1)  # Shape: (num_particles,)
        # Ensure non-negative for sqrt
        sum_sq_sqrt = np.sqrt(np.maximum(sum_sq, 0))  # Shape: (num_particles,)

        numerator = 1.0 + np.cos(12 * sum_sq_sqrt)
        denominator = 2.0 + 0.5 * sum_sq

        # Avoid division by zero
        denominator[denominator == 0] = np.finfo(float).eps

        fitness_values = -numerator / denominator
        return fitness_values  # Shape: (num_particles,)