import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class PinterFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        """
        Initializes the Pinter benchmark function.

        Args:
            dim (int): The dimension of the problem.
            num_particles (int): The number of particles (used by base class, not directly here).
        """
        super().__init__(dim, num_particles)
        self.bounds = (-10, 10) # Bounds specific to Pinter function

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluates the Pinter function for a given input vector x.

        Args:
            x (np.ndarray): A numpy array representing the position vector (particle's position).

        Returns:
            float: The calculated value of the Pinter function.
        """
        n = len(x)
        if n != self.dim:
            raise ValueError(f"Input vector x has dimension {n}, but function dimension is {self.dim}")

        # Term 1: Sum of j * x_j^2
        sum1 = np.sum([(j + 1) * x[j] ** 2 for j in range(n)])

        # Term 2: Sum involving sin of A^2
        # Use modulo arithmetic for indices j-1 and j+1 to handle wrap-around
        sum2 = np.sum([
            20 * (j + 1) * np.sin(
                (x[j - 1] * np.sin(x[j]) + np.sin(x[(j + 1) % n])) ** 2
                # Corrected: x[j + 1] changed to x[(j + 1) % n]
                # Note: x[j-1] works because Python's negative indexing handles j=0 case correctly (x[-1] is last element)
            )
            for j in range(n)
        ])

        # Term 3: Sum involving log10 of (1 + j * B^2)
        # Use modulo arithmetic for indices j-1 and j+1
        # Note: The original code already had modulo for (j+1) here, which was correct.
        # We keep using x[j-1] as negative indexing handles the wrap-around for the start of the array.
        sum3 = np.sum([
            (j + 1) * np.log10(
                1 + (j + 1) * (
                    (x[j - 1] ** 2 - 2 * x[j] + 3 * x[(j + 1) % n] - np.cos(x[j]) + 1) ** 2
                )
            )
            for j in range(n)
        ])

        # Combine the terms
        return sum1 + sum2 + sum3

    # --- Method for PinterFunction ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/Pinter.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Pinter function. """
        # x_matrix shape: (num_particles, dim)
        num_particles, dim = x_matrix.shape
        if dim != self.dim:
            pass  # Handle dim mismatch

        j_indices = np.arange(1, self.dim + 1)  # Shape: (dim,)

        # Use np.roll for circular indexing (j-1 and j+1)
        x_jm1 = np.roll(x_matrix, 1, axis=1)  # x[j-1] equivalent
        x_jp1 = np.roll(x_matrix, -1, axis=1)  # x[j+1] equivalent

        # Term 1: Sum of j * x_j^2
        term1_sum = np.sum(j_indices * x_matrix ** 2, axis=1)  # Shape: (num_particles,)

        # Term 2: Sum involving sin of A^2
        A = x_jm1 * np.sin(x_matrix) + np.sin(x_jp1)  # Shape: (num_particles, dim)
        term2_sum = np.sum(20.0 * j_indices * np.sin(A ** 2), axis=1)  # Shape: (num_particles,)

        # Term 3: Sum involving log10 of (1 + j * B^2)
        B = x_jm1 ** 2 - 2.0 * x_matrix + 3.0 * x_jp1 - np.cos(x_matrix) + 1.0  # Shape: (num_particles, dim)
        log_arg = 1.0 + j_indices * B ** 2
        # Ensure argument for log10 is positive
        log_arg = np.maximum(log_arg, np.finfo(float).eps)  # Replace non-positive with small value
        term3_sum = np.sum(j_indices * np.log10(log_arg), axis=1)  # Shape: (num_particles,)

        fitness_values = term1_sum + term2_sum + term3_sum
        return fitness_values  # Shape: (num_particles,)
