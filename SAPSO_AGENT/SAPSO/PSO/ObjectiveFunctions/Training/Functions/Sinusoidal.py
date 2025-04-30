import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class SinusoidalFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30, A=1, B=1, z=0):
        super().__init__(dim, num_particles)
        self.bounds = (0, 180)
        self.A = A
        self.B = B
        self.z = z

    def evaluate(self, x: np.ndarray) -> float:
        part1 = np.prod(np.sin(x - self.z))
        part2 = np.prod(np.sin(self.B * (x - self.z)))
        return -self.A * (part1 + part2)

    # --- Method for SinusoidalFunction ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/Sinusoidal.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Sinusoidal function. """
        # x_matrix shape: (num_particles, dim)
        term_x_minus_z = x_matrix - self.z  # Shape: (num_particles, dim)

        prod1 = np.prod(np.sin(term_x_minus_z), axis=1)  # Shape: (num_particles,)
        prod2 = np.prod(np.sin(self.B * term_x_minus_z), axis=1)  # Shape: (num_particles,)

        fitness_values = -self.A * (prod1 + prod2)
        return fitness_values  # Shape: (num_particles,)