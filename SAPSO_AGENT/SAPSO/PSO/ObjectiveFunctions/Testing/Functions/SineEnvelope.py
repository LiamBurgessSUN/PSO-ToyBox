import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class SineEnvelopeFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-100, 100)

    def evaluate(self, x: np.ndarray) -> float:
        return np.sum([
            0.5 + (np.sin(np.sqrt(x[j] ** 2 + x[j + 1] ** 2)) ** 2 - 0.5) /
            (1 + 0.001 * (x[j] ** 2 + x[j + 1] ** 2)) ** 2
            for j in range(len(x) - 1)
        ])

    # --- Method for SineEnvelopeFunction ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Testing/SineEnvelope.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Sine Envelope function. """
        # x_matrix shape: (num_particles, dim)
        x_j = x_matrix[:, :-1]  # Shape: (num_particles, dim-1)
        x_j_plus_1 = x_matrix[:, 1:]  # Shape: (num_particles, dim-1)

        term_sq_sum = x_j ** 2 + x_j_plus_1 ** 2
        # Avoid sqrt of zero or negative if bounds allow, though (-100, 100) makes it unlikely for sum
        term_sqrt = np.sqrt(np.maximum(term_sq_sum, 0))  # Ensure non-negative argument for sqrt

        numerator = np.sin(term_sqrt) ** 2 - 0.5
        denominator = (1 + 0.001 * term_sq_sum) ** 2

        # Calculate sum across the dimension pairs for each particle
        fitness_values = np.sum(0.5 + numerator / denominator, axis=1)  # Shape: (num_particles,)
        return fitness_values