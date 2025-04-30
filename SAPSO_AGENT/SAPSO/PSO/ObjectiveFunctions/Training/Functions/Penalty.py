import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction

class Penalty1Function(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30, a=10, k=100, m=4):
        super().__init__(dim, num_particles)
        self.bounds = (-50, 50)
        self.a = a
        self.k = k
        self.m = m

    def u(self, xj):
        if xj > self.a:
            return self.k * (xj - self.a) ** self.m
        elif xj < -self.a:
            return self.k * (-xj - self.a) ** self.m
        return 0

    # --- Methods for Penalty1Function ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/Penalty.py
    # Vectorized helper function u
    def _u_vectorized(self, x_vec: np.ndarray) -> np.ndarray:
        """ Vectorized version of the u penalty function. """
        # x_vec shape: (num_particles,) or (num_particles, dim)
        term1 = self.k * (x_vec - self.a) ** self.m
        term2 = self.k * (-x_vec - self.a) ** self.m

        result = np.where(x_vec > self.a, term1, 0.0)
        result = np.where(x_vec < -self.a, term2, result)
        return result

    def evaluate(self, x: np.ndarray) -> float:
        y = 1 + 0.25 * (x + 1)
        term1 = 10 * np.sin(np.pi * y[0]) ** 2
        term2 = np.sum((y[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * y[1:]) ** 2))
        term3 = (y[-1] - 1) ** 2
        penalty = np.sum([self.u(xj) for xj in x])
        return (np.pi / self.dim) * (term1 + term2 + term3) + penalty

    # Add this evaluate_matrix method to Penalty1Function class
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Penalty N. 1 function. """
        # x_matrix shape: (num_particles, dim)
        num_particles, dim = x_matrix.shape
        if dim != self.dim:
            pass  # Handle dim mismatch

        # Calculate y matrix
        y = 1.0 + 0.25 * (x_matrix + 1.0)  # Shape: (num_particles, dim)

        y_1 = y[:, 0]  # Shape: (num_particles,)
        y_j = y[:, :-1]  # Shape: (num_particles, dim-1)
        y_j_plus_1 = y[:, 1:]  # Shape: (num_particles, dim-1)
        y_n = y[:, -1]  # Shape: (num_particles,)

        term1 = 10.0 * np.sin(np.pi * y_1) ** 2  # Shape: (num_particles,)
        term2_sum = np.sum((y_j - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * y_j_plus_1) ** 2),
                           axis=1)  # Shape: (num_particles,)
        term3 = (y_n - 1.0) ** 2  # Shape: (num_particles,)

        # Calculate penalty term using vectorized u function
        penalty = np.sum(self._u_vectorized(x_matrix), axis=1)  # Shape: (num_particles,)

        fitness_values = (np.pi / self.dim) * (term1 + term2_sum + term3) + penalty
        return fitness_values  # Shape: (num_particles,)

class Penalty2Function(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30, a=5, k=100, m=4):
        super().__init__(dim, num_particles)
        self.bounds = (-50, 50)
        self.a = a
        self.k = k
        self.m = m

    def u(self, xj):
        if xj > self.a:
            return self.k * (xj - self.a) ** self.m
        elif xj < -self.a:
            return self.k * (-xj - self.a) ** self.m
        return 0

    # --- Methods for Penalty1Function ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/Penalty.py
    # Vectorized helper function u
    def _u_vectorized(self, x_vec: np.ndarray) -> np.ndarray:
        """ Vectorized version of the u penalty function. """
        # x_vec shape: (num_particles,) or (num_particles, dim)
        term1 = self.k * (x_vec - self.a) ** self.m
        term2 = self.k * (-x_vec - self.a) ** self.m

        result = np.where(x_vec > self.a, term1, 0.0)
        result = np.where(x_vec < -self.a, term2, result)
        return result

    def evaluate(self, x: np.ndarray) -> float:
        term1 = 0.1 * np.sin(3 * np.pi * x[0]) ** 2
        term2 = np.sum((x[:-1] - 1) ** 2 * (1 + np.sin(3 * np.pi * x[1:]) ** 2))
        term3 = (x[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * x[-1]) ** 2)
        penalty = np.sum([self.u(xj) for xj in x])
        return 0.1 * (term1 + term2 + term3) + penalty

    # --- Methods for Penalty2Function ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/Penalty.py
    # Add this evaluate_matrix method to Penalty2Function class
    # Note: It reuses the _u_vectorized helper defined above for Penalty1
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Penalty N. 2 function. """
        # x_matrix shape: (num_particles, dim)
        num_particles, dim = x_matrix.shape
        if dim != self.dim:
            pass  # Handle dim mismatch

        x_1 = x_matrix[:, 0]  # Shape: (num_particles,)
        x_j = x_matrix[:, :-1]  # Shape: (num_particles, dim-1)
        x_j_plus_1 = x_matrix[:, 1:]  # Shape: (num_particles, dim-1)
        x_n = x_matrix[:, -1]  # Shape: (num_particles,)

        term1 = np.sin(3.0 * np.pi * x_1) ** 2  # Shape: (num_particles,)
        term2_sum = np.sum((x_j - 1.0) ** 2 * (1.0 + np.sin(3.0 * np.pi * x_j_plus_1) ** 2),
                           axis=1)  # Shape: (num_particles,)
        term3 = (x_n - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * x_n) ** 2)  # Shape: (num_particles,)

        # Calculate penalty term using vectorized u function
        # Assumes _u_vectorized is available in the class (defined above for Penalty1)
        # If classes are separate, copy _u_vectorized or make it accessible.
        penalty = np.sum(self._u_vectorized(x_matrix), axis=1)  # Shape: (num_particles,)

        fitness_values = 0.1 * (term1 + term2_sum + term3) + penalty
        return fitness_values  # Shape: (num_particles,)