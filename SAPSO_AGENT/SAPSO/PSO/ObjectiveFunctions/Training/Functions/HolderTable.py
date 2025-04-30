import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class HolderTable1Function(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-10, 10)

    def evaluate(self, x: np.ndarray) -> float:
        prod_cos = np.prod(np.cos(x))
        sum_sq = np.sum(x ** 2)
        return -np.abs(prod_cos * np.exp(np.abs(1 - sum_sq ** 0.5 / np.pi)))

    # --- Method for HolderTable1Function ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/HolderTable.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Holder Table 1 function. """
        # x_matrix shape: (num_particles, dim)
        prod_cos = np.prod(np.cos(x_matrix), axis=1)  # Shape: (num_particles,)
        sum_sq = np.sum(x_matrix ** 2, axis=1)  # Shape: (num_particles,)
        # Ensure non-negative for sqrt
        sum_sq_sqrt = np.sqrt(np.maximum(sum_sq, 0))  # Shape: (num_particles,)

        exp_term = np.exp(np.clip(np.abs(1.0 - sum_sq_sqrt / np.pi), -700, 700))  # Clip exp input

        fitness_values = -np.abs(prod_cos * exp_term)
        return fitness_values  # Shape: (num_particles,)