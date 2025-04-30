import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class NorwegianFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-1.1, 1.1)

    def evaluate(self, x: np.ndarray) -> float:
        return np.prod(np.cos(np.pi * x ** 3) * ((99 + x) / 100))

    # --- Method for NorwegianFunction ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/Norweigan.py (Note typo in filename)
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Norwegian function. """
        # x_matrix shape: (num_particles, dim)
        term1 = np.cos(np.pi * x_matrix ** 3)
        term2 = (99.0 + x_matrix) / 100.0
        # Calculate product along the dimension axis for each particle
        fitness_values = np.prod(term1 * term2, axis=1)  # Shape: (num_particles,)
        return fitness_values