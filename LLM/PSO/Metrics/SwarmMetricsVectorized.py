# File: PSO-ToyBox/LLM/SwarmMetricsVectorized.py
import numpy as np

class SwarmMetricsVectorized:
    """
    Calculates swarm metrics using vectorized NumPy operations.
    Designed to work with PSO implementations that store swarm state in NumPy arrays.
    """
    def __init__(self, stability_threshold=1e-3):
        """
        Initializes the vectorized metrics calculator.

        Args:
            stability_threshold (float): Velocity magnitude below which a particle
                                         is considered stable.
        """
        self.stability_threshold = stability_threshold
        # You could potentially register specific metric functions here if needed,
        # similar to the original SwarmMetrics, but for simplicity,
        # the compute method calculates a fixed set of common metrics.

    def compute(self, positions: np.ndarray, velocities: np.ndarray, bounds: tuple) -> dict:
        """
        Computes various swarm metrics from the provided state arrays.

        Args:
            positions (np.ndarray): Particle positions (num_particles x dim).
            velocities (np.ndarray): Particle velocities (num_particles x dim).
            bounds (tuple): Search space bounds (lower, upper).

        Returns:
            dict: A dictionary containing calculated metrics:
                  'avg_velocity_magnitude', 'stability_ratio',
                  'swarm_diversity', 'feasible_ratio'.
        """
        if positions.shape[0] == 0 or velocities.shape[0] == 0:
             # Handle empty swarm case
             return {
                 'avg_velocity_magnitude': 0.0,
                 'stability_ratio': 1.0, # Or 0.0 depending on definition
                 'swarm_diversity': 0.0,
                 'feasible_ratio': 1.0
             }

        num_particles = positions.shape[0]
        metrics = {}

        # 1. Average Velocity Magnitude (Vectorized)
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        metrics['avg_velocity_magnitude'] = np.mean(velocity_magnitudes)

        # 2. Stability Ratio (Vectorized)
        stable_count = np.sum(velocity_magnitudes < self.stability_threshold)
        metrics['stability_ratio'] = stable_count / num_particles

        # 3. Swarm Diversity (Vectorized)
        if num_particles > 1:
            centroid = np.mean(positions, axis=0) # Shape (dim,)
            # Calculate Euclidean distance from each particle to the centroid
            distances = np.linalg.norm(positions - centroid, axis=1) # Shape (num_particles,)
            metrics['swarm_diversity'] = np.mean(distances)
        else:
            metrics['swarm_diversity'] = 0.0 # Diversity is zero for a single particle

        # 4. Feasibility Ratio (Vectorized)
        # NOTE: If using PSOVectorized which clips positions, particles are *always*
        # within bounds after the position update. This metric might always be 1.0.
        # If feasibility needs to check *before* clipping, the logic would differ.
        lower_bound, upper_bound = bounds
        in_bounds = np.all((positions >= lower_bound) & (positions <= upper_bound), axis=1)
        feasible_count = np.sum(in_bounds)
        metrics['feasible_ratio'] = feasible_count / num_particles

        return metrics

# # === Example Usage ===
# if __name__ == "__main__":
#     # --- Create Dummy Data (mimicking PSOVectorized state) ---
#     N_PARTICLES = 50
#     DIMENSIONS = 10
#     LOWER_BOUND = -10
#     UPPER_BOUND = 10
#     BOUNDS = (LOWER_BOUND, UPPER_BOUND)
#
#     # Random positions within bounds
#     dummy_positions = np.random.uniform(LOWER_BOUND, UPPER_BOUND, size=(N_PARTICLES, DIMENSIONS))
#     # Random velocities (some small, some large)
#     dummy_velocities = np.random.randn(N_PARTICLES, DIMENSIONS) * 0.5 # Mostly small velocities
#     dummy_velocities[0] *= 100 # Make one large
#     dummy_velocities[-1] *= 0.0001 # Make one very small (stable)
#
#     # --- Initialize and Use Metrics Calculator ---
#     metrics_calculator = SwarmMetricsVectorized(stability_threshold=1e-3)
#     calculated_metrics = metrics_calculator.compute(dummy_positions, dummy_velocities, BOUNDS)
#
#     # --- Print Results ---
#     print("--- Calculated Swarm Metrics ---")
#     for key, value in calculated_metrics.items():
#         print(f"{key:<25}: {value:.6f}")
#
#     # Example: Check feasibility if some particles were outside bounds before clipping
#     # dummy_positions[1] = LOWER_BOUND - 1 # Make one particle infeasible
#     # calculated_metrics_infeasible = metrics_calculator.compute(dummy_positions, dummy_velocities, BOUNDS)
#     # print("\n--- Metrics with one infeasible particle (pre-clipping) ---")
#     # print(f"feasible_ratio          : {calculated_metrics_infeasible['feasible_ratio']:.6f}")

