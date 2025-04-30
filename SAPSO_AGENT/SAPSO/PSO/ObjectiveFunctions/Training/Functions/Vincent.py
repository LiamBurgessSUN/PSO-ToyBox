import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class VincentFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (0.25, 10)

    def evaluate(self, x: np.ndarray) -> float:
        return -np.sum(np.sin(10 * np.log(x)))

    # --- Method for VincentFunction ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/Vincent.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Vincent function. """
        # x_matrix shape: (num_particles, dim)
        # Bounds (0.25, 10) ensure x_matrix > 0, so log is safe.
        log_term = np.log(x_matrix)  # Element-wise log
        sin_term = np.sin(10.0 * log_term)  # Element-wise sin
        # Sum across dimensions for each particle
        fitness_values = -np.sum(sin_term, axis=1)  # Shape: (num_particles,)
        return fitness_values
