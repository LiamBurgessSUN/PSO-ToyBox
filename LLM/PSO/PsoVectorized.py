# File: PSO-ToyBox/LLM/PSO/PsoVectorized.py
# Refactored to use the logger module

import numpy as np
import collections
import time # Optional: for basic timing comparison
import traceback # For logging exceptions
from pathlib import Path # To get module name

# --- Import Logger ---
# Using the specified import path: from LLM.Logs import logger
try:
    # Import the module first if needed, then specific functions
    from LLM.Logs import logger
    from LLM.Logs.logger import log_info, log_error, log_warning, log_success, log_header, log_debug
except ImportError:
    # Fallback print if logger fails to import
    print("ERROR: Logger module not found at 'LLM.Logs.logger'. Please check path.")
    print("Falling back to standard print statements.")
    # Define dummy functions
    def log_info(msg, mod): print(f"INFO [{mod}]: {msg}")
    def log_error(msg, mod): print(f"ERROR [{mod}]: {msg}")
    def log_warning(msg, mod): print(f"WARNING [{mod}]: {msg}")
    def log_success(msg, mod): print(f"SUCCESS [{mod}]: {msg}")
    def log_header(msg, mod): print(f"HEADER [{mod}]: {msg}")
    def log_debug(msg, mod): print(f"DEBUG [{mod}]: {msg}") # Optional debug

# --- Project Imports ---
# Import the vectorized metrics calculator
# Assuming it's saved in LLM/PSO/Metrics/SwarmMetricsVectorized.py
try:
    from LLM.PSO.Metrics.SwarmMetricsVectorized import SwarmMetricsVectorized
    # Assuming ObjectiveFunction and KnowledgeSharingStrategy base classes exist
    from LLM.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction
    from LLM.PSO.Cognitive.PositionSharing import KnowledgeSharingStrategy
except ImportError as e:
    # Use fallback print if logger failed during import
    print(f"ERROR [PsoVectorized Setup]: Failed to import necessary PSO modules: {e}")
    print(f"ERROR [PsoVectorized Setup]: Ensure Metrics, ObjectiveFunctions, Cognitive modules are accessible.")
    # Define dummy classes if needed to prevent NameErrors later, or re-raise/exit
    class SwarmMetricsVectorized: pass
    class ObjectiveFunction: pass
    class KnowledgeSharingStrategy: pass


# --- Module Name for Logging ---
module_name = Path(__file__).stem # Gets 'PsoVectorized'

# NOTE: This implementation implicitly uses G-Best strategy due to vectorization simplicity.

class PSOVectorized:
    """
    A vectorized implementation of Particle Swarm Optimization using NumPy arrays.
    Uses an external SwarmMetricsVectorized class for metric calculations.
    Dynamically uses objective_function.evaluate_matrix if available.

    Attributes:
        objective_function: The function to minimize (should have evaluate, optionally evaluate_matrix).
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
                 v_clamp_ratio=0.2,
                 use_velocity_clamping=True,
                 convergence_patience=50,
                 convergence_threshold_gbest=1e-8,
                 convergence_threshold_pbest_std=1e-6,
                 stability_threshold=1e-3 # Threshold for metrics calc
                 ):
        """
        Initializes the vectorized PSO algorithm.
        """
        self.objective_function = objective_function
        self.num_particles = num_particles
        self.dim = objective_function.dim
        self.bounds = objective_function.bounds
        self.kb_sharing_strat = strategy # Kept for compatibility

        # --- Initialize Swarm State with NumPy Arrays ---
        self.positions = np.random.uniform(low=self.bounds[0], high=self.bounds[1], size=(self.num_particles, self.dim))
        self.velocities = np.zeros((self.num_particles, self.dim))

        # Initialize personal bests using the standard evaluate method
        self.pbest_positions = self.positions.copy()
        try:
            # Try vectorized evaluation first if available
            if hasattr(self.objective_function, 'evaluate_matrix') and callable(self.objective_function.evaluate_matrix):
                 self.pbest_values = self.objective_function.evaluate_matrix(self.positions)
                 log_debug("Initialized pbest using evaluate_matrix.", module_name)
            else:
                 log_debug("evaluate_matrix not found, initializing pbest using loop.", module_name)
                 self.pbest_values = np.array([self.objective_function.evaluate(p) for p in self.positions])
        except Exception as e:
            log_error(f"Error during initial pbest evaluation: {e}. Initializing pbest values to infinity.", module_name)
            log_error(traceback.format_exc(), module_name)
            self.pbest_values = np.full(self.num_particles, np.inf)

        if not np.all(np.isfinite(self.pbest_values)):
             log_warning("Non-finite initial pbest values detected. Replacing with inf.", module_name)
             self.pbest_values[~np.isfinite(self.pbest_values)] = np.inf

        # Initialize global best
        min_idx = np.argmin(self.pbest_values)
        # Ensure min_idx is valid before accessing arrays
        if len(self.pbest_values) > 0 and np.isfinite(self.pbest_values[min_idx]):
             self.gbest_position = self.pbest_positions[min_idx].copy()
             self.gbest_value = self.pbest_values[min_idx]
        else:
             # Fallback if initial evaluation failed or resulted in all inf
             self.gbest_position = self.positions[0].copy() if self.num_particles > 0 else np.zeros(self.dim) # Handle zero particles case
             self.gbest_value = np.inf
             log_warning("Could not determine initial gbest from pbest values. Initializing gbest to inf.", module_name)


        # --- Velocity Clamping Setup ---
        self.use_velocity_clamping = use_velocity_clamping
        range_width = self.bounds[1] - self.bounds[0]
        self.v_max = v_clamp_ratio * range_width if range_width > 0 else v_clamp_ratio # Calculate v_max

        # --- Convergence Tracking ---
        self.convergence_patience = convergence_patience
        self.convergence_threshold_gbest = convergence_threshold_gbest
        self.convergence_threshold_pbest_std = convergence_threshold_pbest_std
        self._gbest_history = collections.deque(maxlen=convergence_patience)
        self._stagnation_counter = 0
        self.reset_convergence_tracking() # Initialize history

        # --- Instantiate Metrics Calculator ---
        try:
            self.metrics_calculator = SwarmMetricsVectorized(stability_threshold=stability_threshold)
        except NameError:
             log_error("SwarmMetricsVectorized class not found. Metrics calculation disabled.", module_name)
             self.metrics_calculator = None # Disable metrics

        log_info(f"Initialized Vectorized PSO: {num_particles} particles, {self.dim} dimensions.", module_name)
        log_info(f"Initial GBest Value: {self.gbest_value:.4e}", module_name)

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
        # Vectorized PSO typically uses G-Best implicitly
        social_velocity = c2 * r2 * (self.gbest_position - self.positions) # gbest_position broadcasts
        inertia_velocity = omega * self.velocities

        # --- Update Velocities ---
        self.velocities = inertia_velocity + cognitive_velocity + social_velocity

        # --- Apply Velocity Clamping (Vectorized) ---
        if self.use_velocity_clamping:
            # Ensure v_max is scalar or broadcastable
            self.velocities = np.clip(self.velocities, -self.v_max, self.v_max)

        # --- Update Positions (Vectorized) ---
        self.positions += self.velocities

        # --- Apply Boundary Constraints (Vectorized) ---
        self.positions = np.clip(self.positions, self.bounds[0], self.bounds[1])

        # --- Evaluate New Positions ---
        fitness_values = np.full(self.num_particles, np.inf) # Default to inf
        try:
            # Check if the objective function has a vectorized 'evaluate_matrix' method
            if hasattr(self.objective_function, 'evaluate_matrix') and callable(self.objective_function.evaluate_matrix):
                fitness_values = self.objective_function.evaluate_matrix(self.positions)
                # Basic check for expected output shape
                if fitness_values.shape != (self.num_particles,):
                     log_warning(f"evaluate_matrix returned unexpected shape {fitness_values.shape}. Expected ({self.num_particles},). Falling back to loop.", module_name)
                     fitness_values = np.array([self.objective_function.evaluate(p) for p in self.positions])
            else:
                # Fallback: Loop if evaluate_matrix is not available
                log_debug("evaluate_matrix not found, using loop evaluation.", module_name)
                fitness_values = np.array([self.objective_function.evaluate(p) for p in self.positions])
        except Exception as e:
            log_error(f"Error during vectorized evaluation: {e}. Setting fitness to infinity.", module_name)
            log_error(traceback.format_exc(), module_name)
            # fitness_values remains np.inf initialized above

        # Handle potential non-finite values from evaluation
        if not np.all(np.isfinite(fitness_values)):
             log_warning("Non-finite fitness values detected during evaluation. Replacing with inf.", module_name)
             fitness_values[~np.isfinite(fitness_values)] = np.inf

        # --- Update Personal Bests (Vectorized) ---
        improvement_mask = fitness_values < self.pbest_values
        self.pbest_positions[improvement_mask] = self.positions[improvement_mask]
        self.pbest_values[improvement_mask] = fitness_values[improvement_mask]

        # --- Update Global Best ---
        current_min_idx = np.argmin(self.pbest_values)
        # Check if the best value is actually finite before updating
        if np.isfinite(self.pbest_values[current_min_idx]):
             current_gbest_value = self.pbest_values[current_min_idx]
             if current_gbest_value < self.gbest_value:
                 self.gbest_value = current_gbest_value
                 self.gbest_position = self.pbest_positions[current_min_idx].copy()
        # else: gbest remains unchanged if all pbest values are infinite

        # --- Track GBest History for Stagnation Check ---
        # Ensure only finite values are added to history if possible
        value_to_add = self.gbest_value if np.isfinite(self.gbest_value) else self._gbest_history[-1] if self._gbest_history else float('inf')
        self._gbest_history.append(value_to_add)

        # --- Calculate Metrics using External Calculator ---
        metrics = {}
        if self.metrics_calculator:
            try:
                metrics = self.metrics_calculator.compute(self.positions, self.velocities, self.bounds)
            except Exception as e:
                log_error(f"Error computing metrics: {e}", module_name)
                # metrics remains empty
        # Add gbest_value to metrics dict for convenience (often needed together)
        metrics['gbest_value'] = self.gbest_value

        # --- Check for Swarm Convergence ---
        converged = self._check_convergence()

        return metrics, self.gbest_value, converged

    def _check_convergence(self) -> bool:
        """
        Checks if the swarm has converged based on diversity and gbest stagnation.
        (Adapted for vectorized data).
        """
        # 1. Check GBest Stagnation
        gbest_stagnated = False
        if len(self._gbest_history) == self.convergence_patience:
            history_start_val = self._gbest_history[0]
            if np.isfinite(history_start_val) and np.isfinite(self.gbest_value):
                 improvement = history_start_val - self.gbest_value
                 if improvement < self.convergence_threshold_gbest and improvement >= 0: # Check improvement >= 0
                     self._stagnation_counter += 1
                 else:
                     self._stagnation_counter = 0
            else:
                 self._stagnation_counter = 0
                 log_debug("Non-finite gbest value in history, resetting stagnation counter.", module_name)

            if self._stagnation_counter >= self.convergence_patience:
                 gbest_stagnated = True
                 log_debug(f"GBest stagnation detected ({self._stagnation_counter} steps).", module_name)
        else:
             self._stagnation_counter = 0

        # 2. Check Swarm Diversity (using std dev of pbest values)
        finite_pbest_values = self.pbest_values[np.isfinite(self.pbest_values)]
        diversity_low = False
        pbest_std_dev = np.nan

        if len(finite_pbest_values) < 2:
            diversity_low = gbest_stagnated # Consider non-diverse only if gbest also stagnated
            log_debug(f"Diversity check: Not enough finite pbest values ({len(finite_pbest_values)}). diversity_low set to {diversity_low}.", module_name)
        else:
            pbest_std_dev = np.std(finite_pbest_values)
            diversity_low = pbest_std_dev < self.convergence_threshold_pbest_std
            log_debug(f"Diversity check: PBest Std Dev = {pbest_std_dev:.2e}. Threshold = {self.convergence_threshold_pbest_std:.2e}. diversity_low = {diversity_low}", module_name)

        # 3. Combine Conditions
        if gbest_stagnated and diversity_low:
            log_debug(f"Convergence detected: GBest Stagnated ({self._stagnation_counter} steps) & PBest Std Dev ({pbest_std_dev:.2e}) low.", module_name)
            return True

        return False

    def reset_convergence_tracking(self):
        """Resets gbest history and stagnation counter."""
        self._gbest_history.clear()
        initial_gbest = self.gbest_value if hasattr(self, 'gbest_value') and np.isfinite(self.gbest_value) else float('inf')
        self._gbest_history.append(initial_gbest)
        self._stagnation_counter = 0
        log_debug("Convergence tracking reset.", module_name)

