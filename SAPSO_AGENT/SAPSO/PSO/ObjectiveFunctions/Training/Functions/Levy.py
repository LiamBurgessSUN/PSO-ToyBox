import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class Levy3Function(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-10, 10)

    def evaluate(self, x: np.ndarray) -> float:
        y = 1 + (x - 1) / 4
        term1 = np.sin(np.pi * y[0]) ** 2
        term2 = np.sum((y[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * y[1:]) ** 2))
        term3 = (y[-1] - 1) ** 2
        return term1 + term2 + term3

    # --- Method for Levy3Function ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/Levy.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Levy N. 3 function. """
        # x_matrix shape: (num_particles, dim)
        # Calculate y matrix first
        y = 1.0 + (x_matrix - 1.0) / 4.0  # Shape: (num_particles, dim)

        y_j = y[:, :-1]  # Shape: (num_particles, dim-1)
        y_j_plus_1 = y[:, 1:]  # Shape: (num_particles, dim-1)
        y_n = y[:, -1]  # Last element for term3, shape: (num_particles,)
        y_1 = y[:, 0]  # First element for term1, shape: (num_particles,)

        term1 = np.sin(np.pi * y_1) ** 2  # Shape: (num_particles,)
        term2_sum = np.sum((y_j - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * y_j_plus_1) ** 2),
                           axis=1)  # Shape: (num_particles,)
        term3 = (y_n - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * y_n) ** 2)  # Term 3 in original Levy (not Levy N.13)
        # The code implements Levy N.13, which has a simpler term3:
        term3_levy13 = (y_n - 1) ** 2  # Shape: (num_particles,)

        # Using the formula from the file (Levy N.13)
        fitness_values = term1 + term2_sum + term3_levy13
        return fitness_values  # Shape: (num_particles,)