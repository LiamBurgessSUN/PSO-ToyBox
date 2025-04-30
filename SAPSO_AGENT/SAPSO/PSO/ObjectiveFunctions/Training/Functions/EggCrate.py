import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class EggCrateFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-5, 5)

    def evaluate(self, x: np.ndarray) -> float:
        return np.sum(x ** 2) + 24 * np.sum(np.sin(x) ** 2)

    # --- Method for EggCrateFunction ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/EggCrate.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Egg Crate function. """
        # x_matrix shape: (num_particles, dim)
        term1 = np.sum(x_matrix ** 2, axis=1)  # Shape: (num_particles,)
        term2 = 24 * np.sum(np.sin(x_matrix) ** 2, axis=1)  # Shape: (num_particles,)
        fitness_values = term1 + term2
        return fitness_values