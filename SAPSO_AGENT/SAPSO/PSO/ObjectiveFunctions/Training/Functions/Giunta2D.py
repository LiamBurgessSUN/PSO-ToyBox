import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class GiuntaFunction(ObjectiveFunction):
    def __init__(self, dim=2, num_particles=30):  # Must be 2D
        assert dim == 2, "Giunta function is only defined for 2D."
        super().__init__(dim, num_particles)
        self.bounds = (-1, 1)

    def evaluate(self, x: np.ndarray) -> float:
        val = 0.6
        for j in range(2):
            a = (16 / 15) * x[j] - 1
            val += np.sin(a) + np.sin(a) ** 2 + (1/50) * np.sin(4 * a)
        return val

    # --- Method for GiuntaFunction ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/Giunta2D.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Giunta function (must be 2D). """
        # x_matrix shape: (num_particles, dim)
        num_particles, dim = x_matrix.shape
        assert dim == 2, "Giunta function is only defined for 2D."
        if dim != self.dim:  # self.dim should also be 2
            pass

        # Calculate term a for all particles and both dimensions
        a = (16.0 / 15.0) * x_matrix - 1.0  # Shape: (num_particles, 2)

        # Calculate the sum part for all particles and dimensions
        sum_term = np.sin(a) + np.sin(a) ** 2 + (1.0 / 50.0) * np.sin(4.0 * a)  # Shape: (num_particles, 2)

        # Sum across the 2 dimensions for each particle and add constant
        fitness_values = 0.6 + np.sum(sum_term, axis=1)  # Shape: (num_particles,)
        return fitness_values
