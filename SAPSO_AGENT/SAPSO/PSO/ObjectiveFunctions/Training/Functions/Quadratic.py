import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class QuadricFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-100, 100)

    def evaluate(self, x: np.ndarray) -> float:
        return np.sum([np.sum(x[:j + 1]) ** 2 for j in range(len(x))])

    # --- Method for QuadricFunction ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/Quadratic.py (Note filename typo)
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Quadric function. """
        # x_matrix shape: (num_particles, dim)
        # Calculate cumulative sum along the dimension axis for each particle
        # cumsum[i, j] = sum(x[i, 0] to x[i, j])
        inner_sums = np.cumsum(x_matrix, axis=1)  # Shape: (num_particles, dim)

        # Square the cumulative sums and sum across dimensions for each particle
        fitness_values = np.sum(inner_sums ** 2, axis=1)  # Shape: (num_particles,)
        return fitness_values