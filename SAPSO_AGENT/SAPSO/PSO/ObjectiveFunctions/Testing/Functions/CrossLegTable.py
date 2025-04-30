import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class CrossLegTableFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-10, 10)

    def evaluate(self, x: np.ndarray) -> float:
        norm = np.sqrt(np.sum(x ** 2))
        sin_prod = np.abs(np.prod(np.sin(x)))
        denominator = np.abs(np.exp(np.abs(100 - norm / np.pi)) * (sin_prod + 1))
        return 1 / (denominator ** 0.1)

    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for CrossLegTable function. """
        # x_matrix shape: (num_particles, dim)
        norms = np.sqrt(np.sum(x_matrix ** 2, axis=1)) # Shape: (num_particles,)
        sin_prods = np.abs(np.prod(np.sin(x_matrix), axis=1)) # Shape: (num_particles,)

        # Calculate denominator, handle potential division by zero or overflow in exp
        exp_term = np.exp(np.clip(np.abs(100 - norms / np.pi), -700, 700)) # Clip exp input to avoid overflow
        denominators = np.abs(exp_term * (sin_prods + 1))

        # Avoid division by zero
        denominators[denominators == 0] = np.finfo(float).eps # Replace 0 with a small number

        fitness_values = -1 / (denominators ** 0.1) # Note: Original paper has -1 / ..., maximizing this is minimizing the original
        # If minimizing, use: fitness_values = 1 / (denominators ** 0.1)
        # Sticking to the formula in the provided file:
        fitness_values = 1 / (denominators ** 0.1)

        return fitness_values # Shape: (num_particles,)