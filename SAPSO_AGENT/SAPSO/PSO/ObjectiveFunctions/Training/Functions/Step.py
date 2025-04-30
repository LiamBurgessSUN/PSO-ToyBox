import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class StepFunction3(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-5.12, 5.12)

    def evaluate(self, x: np.ndarray) -> float:
        return np.sum(np.floor(np.abs(x)))

    # --- Method for StepFunction3 ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/Step.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Step N. 3 function. """
        # x_matrix shape: (num_particles, dim)
        # np.floor works element-wise
        term = np.floor(np.abs(x_matrix))  # Shape: (num_particles, dim)
        fitness_values = np.sum(term, axis=1)  # Shape: (num_particles,)
        return fitness_values