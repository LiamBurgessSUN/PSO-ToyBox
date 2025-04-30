import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class NeedleEyeFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30, eye=0.0001):
        super().__init__(dim, num_particles)
        self.bounds = (-10, 10)
        self.eye = eye

    def evaluate(self, x: np.ndarray) -> float:
        if np.all(np.abs(x) < self.eye):
            return 1
        elif np.all(np.abs(x) <= self.eye):
            return np.sum(100 + np.abs(x))
        else:
            return 0

    # --- Method for NeedleEyeFunction ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/NeedleEye.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Needle Eye function. """
        # x_matrix shape: (num_particles, dim)
        num_particles = x_matrix.shape[0]
        fitness_values = np.zeros(num_particles)

        abs_x = np.abs(x_matrix)

        # Condition 1: all |xj| < eye
        cond1_mask = np.all(abs_x < self.eye, axis=1)
        fitness_values[cond1_mask] = 1.0

        # Condition 2: all |xj| <= eye (and not condition 1)
        # This condition seems redundant or maybe meant something else?
        # The original code has: elif np.all(np.abs(x) <= self.eye): return np.sum(100 + np.abs(x))
        # This elif is only reached if the first if (all < eye) is false.
        # So it means: at least one |xj| == eye, and all others are <= eye.
        # Let's implement based on the likely *intent* which might be:
        # If *any* |xj| > eye, return 0, otherwise apply the sum or 1.

        # Alternative Interpretation based on code structure:
        # Check if *any* element is outside the eye threshold
        cond3_mask = np.any(abs_x > self.eye, axis=1)
        fitness_values[cond3_mask] = 0.0

        # Condition 2 (Implicitly): Not Cond1 and Not Cond3 means all <= eye and at least one == eye
        # Or maybe the original code intended: If any |xj| is > eye, return 0. If all |xj| < eye, return 1. Otherwise (all |xj| <= eye and at least one == eye), return sum.
        cond2_mask = ~cond1_mask & ~cond3_mask
        if np.any(cond2_mask):
            sum_term = np.sum(100.0 + abs_x[cond2_mask], axis=1)
            fitness_values[cond2_mask] = sum_term

        # Let's re-implement strictly following the original code's logic flow:
        fitness_values = np.zeros(num_particles)  # Reset
        abs_x = np.abs(x_matrix)
        # Check condition 1 first
        cond1_mask = np.all(abs_x < self.eye, axis=1)
        fitness_values[cond1_mask] = 1.0
        # Check condition 2 only for those not meeting condition 1
        remaining_mask = ~cond1_mask
        if np.any(remaining_mask):
            cond2_mask_subset = np.all(abs_x[remaining_mask] <= self.eye, axis=1)
            # Apply condition 2 sum where cond2_mask_subset is True within the remaining_mask
            true_cond2_indices = np.where(remaining_mask)[0][cond2_mask_subset]
            if len(true_cond2_indices) > 0:
                sum_term = np.sum(100.0 + abs_x[true_cond2_indices], axis=1)
                fitness_values[true_cond2_indices] = sum_term
        # Condition 3 (else: return 0) is implicitly handled as values are initialized to 0

        return fitness_values  # Shape: (num_particles,)