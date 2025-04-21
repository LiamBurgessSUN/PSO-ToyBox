# File: PSO-ToyBox/LLM/PSO/PSOVectorized.py
import numpy as np
import collections
import time # Optional: for basic timing comparison

from LLM.PSO.Metrics.SwarmMetricsVectorized import SwarmMetricsVectorized


# Import the vectorized metrics calculator
# Assuming it's saved in LLM.SwarmMetricsVectorized
# If the file path is different, adjust the import accordingly.



# Assuming ObjectiveFunction and KnowledgeSharingStrategy base classes exist
# from LLM.PSO.ObjectiveFunctions.Training.ObjectiveFunction import ObjectiveFunction
# from LLM.PSO.Cognitive.PositionSharing import KnowledgeSharingStrategy
# NOTE: This implementation implicitly uses G-Best strategy due to vectorization simplicity.

class PSOVectorized:
    """
    A vectorized implementation of Particle Swarm Optimization using NumPy arrays.
    Uses an external SwarmMetricsVectorized class for metric calculations.

    Attributes:
        objective_function: The function to minimize.
        num_particles (int): Number of particles in the swarm.
        dim (int): Dimension of the search space.
        bounds (tuple): Lower and upper bounds for particle positions (lower, upper).
        positions (np.ndarray): Current positions of all particles (shape: num_particles x dim).
        velocities (np.ndarray): Current velocities of all particles (shape: num_particles x dim).
        pbest_positions (np.ndarray): Personal best positions found by each particle.
        pbest_values (np.ndarray): Personal best fitness values for each particle.
        gbest_position (np.ndarray): Global best position found by the swarm.
        gbest_value (float): Global best fitness value found by the swarm.
        metrics_calculator (SwarmMetricsVectorized): Instance for calculating metrics.
        use_velocity_clamping (bool): Whether to clamp velocities.
        v_max (float): Maximum velocity component value if clamping is enabled.
        convergence_patience (int): Number of steps gbest must stagnate for convergence.
        convergence_threshold_gbest (float): Max gbest improvement considered stagnation.
        convergence_threshold_pbest_std (float): Max pbest std dev for convergence.
        _gbest_history (deque): Stores recent gbest values for stagnation check.
        _stagnation_counter (int): Counts steps gbest hasn't improved sufficiently.
    """
    def __init__(self, objective_function, num_particles,
                 strategy=None, # Strategy object kept for compatibility
                 v_clamp_ratio=0.2, use_velocity_clamping=True,
                 convergence_patience=50,
                 convergence_threshold_gbest=1e-8,
                 convergence_threshold_pbest_std=1e-6,
                 stability_threshold=1e-3 # Threshold for metrics calc
                 ):
        """
        Initializes the vectorized PSO algorithm.

        Args:
            objective_function: An object with `dim`, `bounds`, and `evaluate` method.
            num_particles (int): The number of particles in the swarm.
            strategy: Knowledge sharing strategy (Note: G-Best assumed here).
            v_clamp_ratio (float): Ratio of search space range for velocity clamping.
            use_velocity_clamping (bool): Flag to enable/disable velocity clamping.
            convergence_patience (int): Steps for gbest stagnation check.
            convergence_threshold_gbest (float): Threshold for gbest improvement.
            convergence_threshold_pbest_std (float): Threshold for pbest standard deviation.
            stability_threshold (float): Velocity threshold for stability metric.
        """
        self.objective_function = objective_function
        self.num_particles = num_particles
        self.dim = objective_function.dim
        self.bounds = objective_function.bounds
        self.kb_sharing_strat = strategy # Kept for compatibility

        # --- Initialize Swarm State with NumPy Arrays ---
        self.positions = np.random.uniform(low=self.bounds[0], high=self.bounds[1], size=(self.num_particles, self.dim))
        self.velocities = np.zeros((self.num_particles, self.dim))

        # Initialize personal bests
        self.pbest_positions = self.positions.copy()
        self.pbest_values = np.array([self.objective_function.evaluate(p) for p in self.positions])
        if not np.all(np.isfinite(self.pbest_values)):
             print("Warning: Non-finite initial pbest values detected. Replacing with inf.")
             self.pbest_values[~np.isfinite(self.pbest_values)] = np.inf

        # Initialize global best
        min_idx = np.argmin(self.pbest_values)
        self.gbest_position = self.pbest_positions[min_idx].copy()
        self.gbest_value = self.pbest_values[min_idx]

        # --- Velocity Clamping Setup ---
        self.use_velocity_clamping = use_velocity_clamping
        self.v_max = v_clamp_ratio * (self.bounds[1] - self.bounds[0])

        # --- Convergence Tracking ---
        self.convergence_patience = convergence_patience
        self.convergence_threshold_gbest = convergence_threshold_gbest
        self.convergence_threshold_pbest_std = convergence_threshold_pbest_std
        self._gbest_history = collections.deque(maxlen=convergence_patience)
        self._stagnation_counter = 0
        self.reset_convergence_tracking() # Initialize history

        # --- Instantiate Metrics Calculator ---
        self.metrics_calculator = SwarmMetricsVectorized(stability_threshold=stability_threshold)

        print(f"Initialized Vectorized PSO: {num_particles} particles, {self.dim} dimensions.")
        print(f"Initial GBest Value: {self.gbest_value:.4e}")

    def optimize_step(self, omega, c1, c2):
        """
        Performs one vectorized step of the PSO optimization.

        Args:
            omega (float): Inertia weight.
            c1 (float): Cognitive coefficient.
            c2 (float): Social coefficient.

        Returns:
            tuple: (metrics, gbest_value, converged)
                   - metrics (dict): Dictionary of swarm metrics for the step.
                   - gbest_value (float): The global best value after the step.
                   - converged (bool): True if swarm convergence criteria met.
        """
        # --- Generate Random Numbers ---
        r1 = np.random.rand(self.num_particles, self.dim)
        r2 = np.random.rand(self.num_particles, self.dim)

        # --- Calculate Velocity Components (Vectorized) ---
        cognitive_velocity = c1 * r1 * (self.pbest_positions - self.positions)
        social_velocity = c2 * r2 * (self.gbest_position - self.positions)
        inertia_velocity = omega * self.velocities

        # --- Update Velocities ---
        self.velocities = inertia_velocity + cognitive_velocity + social_velocity

        # --- Apply Velocity Clamping (Vectorized) ---
        if self.use_velocity_clamping:
            self.velocities = np.clip(self.velocities, -self.v_max, self.v_max)

        # --- Update Positions (Vectorized) ---
        self.positions += self.velocities

        # --- Apply Boundary Constraints (Vectorized) ---
        self.positions = np.clip(self.positions, self.bounds[0], self.bounds[1])

        # --- Evaluate New Positions ---
        try:
            # Fallback: Loop if evaluate expects single vector
            fitness_values = np.array([self.objective_function.evaluate(p) for p in self.positions])
            # NOTE: Replace loop above with below line if objective_function supports matrix evaluation
            # fitness_values = self.objective_function.evaluate(self.positions)
        except Exception as e:
             # Fallback loop if vectorized attempt fails or is not implemented
             fitness_values = np.array([self.objective_function.evaluate(p) for p in self.positions])

        if not np.all(np.isfinite(fitness_values)):
             fitness_values[~np.isfinite(fitness_values)] = np.inf

        # --- Update Personal Bests (Vectorized) ---
        improvement_mask = fitness_values < self.pbest_values
        self.pbest_positions[improvement_mask] = self.positions[improvement_mask]
        self.pbest_values[improvement_mask] = fitness_values[improvement_mask]

        # --- Update Global Best ---
        current_min_idx = np.argmin(self.pbest_values)
        current_gbest_value = self.pbest_values[current_min_idx]

        if current_gbest_value < self.gbest_value:
            self.gbest_value = current_gbest_value
            self.gbest_position = self.pbest_positions[current_min_idx].copy()

        # --- Track GBest History for Stagnation Check ---
        self._gbest_history.append(self.gbest_value)

        # --- Calculate Metrics using External Calculator ---
        metrics = self.metrics_calculator.compute(self.positions, self.velocities, self.bounds)
        # Add gbest_value to metrics dict for convenience (often needed together)
        metrics['gbest_value'] = self.gbest_value

        # --- Check for Swarm Convergence ---
        converged = self._check_convergence()

        return metrics, self.gbest_value, converged

    # Removed internal _calculate_metrics method

    def _check_convergence(self) -> bool:
        """
        Checks if the swarm has converged based on diversity and gbest stagnation.
        (Adapted for vectorized data).
        """
        # 1. Check GBest Stagnation
        gbest_stagnated = False
        if len(self._gbest_history) == self.convergence_patience:
            improvement = self._gbest_history[0] - self.gbest_value
            if improvement < self.convergence_threshold_gbest and not np.isclose(improvement, self.convergence_threshold_gbest):
                 self._stagnation_counter += 1
            else:
                 self._stagnation_counter = 0 # Reset if significant improvement

            if self._stagnation_counter >= self.convergence_patience:
                 gbest_stagnated = True
        else:
             self._stagnation_counter = 0

        # 2. Check Swarm Diversity (using std dev of pbest values)
        finite_pbest_values = self.pbest_values[np.isfinite(self.pbest_values)]
        if len(finite_pbest_values) < 2:
            diversity_low = gbest_stagnated
        else:
            pbest_std_dev = np.std(finite_pbest_values)
            diversity_low = pbest_std_dev < self.convergence_threshold_pbest_std

        # 3. Combine Conditions
        if gbest_stagnated and diversity_low:
            return True

        return False

    def reset_convergence_tracking(self):
        """Resets gbest history and stagnation counter."""
        self._gbest_history.clear()
        initial_gbest = self.gbest_value if hasattr(self, 'gbest_value') and np.isfinite(self.gbest_value) else np.inf
        self._gbest_history.append(initial_gbest)
        self._stagnation_counter = 0

