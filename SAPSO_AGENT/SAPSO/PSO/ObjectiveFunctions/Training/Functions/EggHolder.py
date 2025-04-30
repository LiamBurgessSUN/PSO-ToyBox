import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class EggHolderFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-512, 512)

    def evaluate(self, x: np.ndarray) -> float:
        return np.sum(
            -(x[1:] + 47) * np.sin(np.sqrt(np.abs(x[1:] + x[:-1] / 2 + 47))) -
            x[:-1] * np.sin(np.sqrt(np.abs(x[:-1] - (x[1:] + 47))))
        )

    # --- Method for EggHolderFunction ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/EggHolder.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Egg Holder function. """
        # x_matrix shape: (num_particles, dim)
        x_j = x_matrix[:, :-1]  # Shape: (num_particles, dim-1)
        x_j_plus_1 = x_matrix[:, 1:]  # Shape: (num_particles, dim-1)

        term1_sqrt_arg = np.abs(x_j_plus_1 + x_j / 2.0 + 47.0)
        term1 = -(x_j_plus_1 + 47.0) * np.sin(np.sqrt(term1_sqrt_arg))

        term2_sqrt_arg = np.abs(x_j - (x_j_plus_1 + 47.0))
        term2 = -x_j * np.sin(np.sqrt(term2_sqrt_arg))

        # Sum across the dimension pairs for each particle
        fitness_values = np.sum(term1 + term2, axis=1)  # Shape: (num_particles,)
        return fitness_values