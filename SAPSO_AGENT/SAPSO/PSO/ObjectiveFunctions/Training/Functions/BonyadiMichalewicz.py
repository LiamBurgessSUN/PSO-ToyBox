import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class BonyadiMichalewiczFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-5, 5)

    def evaluate(self, x: np.ndarray) -> float:
        numerator = np.prod(x + 1)
        denominator = np.prod((x + 1) ** 2 + 1)
        return numerator / denominator

    # --- Method for BonyadiMichalewiczFunction ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/BonyadiMichalewicz.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Bonyadi-Michalewicz function. """
        # x_matrix shape: (num_particles, dim)
        term_plus_1 = x_matrix + 1.0

        numerator = np.prod(term_plus_1, axis=1)  # Shape: (num_particles,)
        denominator = np.prod(term_plus_1 ** 2 + 1.0, axis=1)  # Shape: (num_particles,)

        # Avoid division by zero
        denominator[denominator == 0] = np.finfo(float).eps

        fitness_values = numerator / denominator  # Shape: (num_particles,)
        # Assuming minimization, the function is often defined to be maximized.
        # If minimizing, return fitness_values. If maximizing, return -fitness_values.
        # Sticking to formula provided:
        return fitness_values