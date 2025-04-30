import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class BrownFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-1, 1)

    def evaluate(self, x: np.ndarray) -> float:
        return np.sum((x[:-1]**2)**(x[1:]**2 + 1) + (x[1:]**2)**(x[:-1]**2 + 1))

    # --- Method for BrownFunction ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/Brown.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Brown function. """
        # x_matrix shape: (num_particles, dim)
        x_j_sq = x_matrix[:, :-1] ** 2  # Shape: (num_particles, dim-1)
        x_j_plus_1_sq = x_matrix[:, 1:] ** 2  # Shape: (num_particles, dim-1)

        term1 = x_j_sq ** (x_j_plus_1_sq + 1.0)
        term2 = x_j_plus_1_sq ** (x_j_sq + 1.0)

        fitness_values = np.sum(term1 + term2, axis=1)  # Shape: (num_particles,)
        return fitness_values