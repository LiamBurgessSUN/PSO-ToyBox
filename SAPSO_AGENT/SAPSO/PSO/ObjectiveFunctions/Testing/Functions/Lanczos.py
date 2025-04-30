import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class Lanczos3Function(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-20, 20)

    def sinc(self, x: np.ndarray) -> np.ndarray:
        return np.sinc(x / np.pi)

    def evaluate(self, x: np.ndarray) -> float:
        return np.prod(self.sinc(x) * self.sinc(x / 3))

    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Lanczos3 function. """
        # x_matrix shape: (num_particles, dim)

        # Define sinc function (np.sinc includes the pi factor: sin(pi*x)/(pi*x))
        # The formula uses sin(x)/x, so we need to handle the division by pi
        def custom_sinc(x):
            # Handle x=0 case where sinc is 1
            return np.divide(np.sin(x), x, out=np.ones_like(x), where=x!=0)

        sinc_x = custom_sinc(x_matrix)
        sinc_x_over_3 = custom_sinc(x_matrix / 3.0)

        # Calculate product along the dimension axis for each particle
        fitness_values = np.prod(sinc_x * sinc_x_over_3, axis=1) # Shape: (num_particles,)

        # Lanczos is often maximized, PSO minimizes. Return negative if needed.
        # Assuming minimization as per standard PSO benchmarks:
        return fitness_values # Or -fitness_values if maximization is intended