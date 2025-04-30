# --- Rastrigin Function Implementation ---
import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class AckleyFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.dim = dim
        self.num_particles = num_particles
        self.bounds = (-32, 32)  # Default bounds


    def evaluate(self, x: np.ndarray) -> float:
        return -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / self.dim)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / self.dim) + 20 + np.e

    # --- Method for AckleyFunction ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/Ackley.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Ackley function. """
        # x_matrix shape: (num_particles, dim)
        n = self.dim
        term1 = -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x_matrix ** 2, axis=1) / n))
        term2 = -np.exp(np.sum(np.cos(2 * np.pi * x_matrix), axis=1) / n)
        fitness_values = term1 + term2 + 20.0 + np.e
        return fitness_values  # Shape: (num_particles,)