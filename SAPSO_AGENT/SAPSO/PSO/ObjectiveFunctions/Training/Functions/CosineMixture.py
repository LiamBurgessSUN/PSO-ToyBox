import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class CosineMixtureFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-1, 1)

    def evaluate(self, x: np.ndarray) -> float:
        return 0.1 * np.sum(np.cos(5 * np.pi * x)) + np.sum(x ** 2)

    # --- Method for CosineMixtureFunction ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/CosineMixture.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Cosine Mixture function. """
        # x_matrix shape: (num_particles, dim)
        term1 = 0.1 * np.sum(np.cos(5 * np.pi * x_matrix), axis=1)  # Shape: (num_particles,)
        term2 = np.sum(x_matrix ** 2, axis=1)  # Shape: (num_particles,)
        fitness_values = term1 + term2  # Note: Original formula in paper often has -0.1
        # Sticking to formula provided:
        return fitness_values