# File: PSO-ToyBox/SAPSO_AGENT/PSO/PSO.py
# Modified to store previous positions and pass necessary parameters
# to the metrics calculator for paper-aligned metric calculations.
# --- UPDATED optimize_step to handle boundary constraints as per paper (assign inf fitness) ---

import numpy as np
import collections
import traceback  # For logging exceptions
from pathlib import Path  # To get module name
from typing import Dict, Any, Optional, Tuple

from SAPSO_AGENT.Logs.logger import *
from SAPSO_AGENT.SAPSO.PSO.Metrics.Metrics import SwarmMetrics

# --- Module Name for Logging ---
module_name = Path(__file__).stem  # Gets 'PsoVectorized'


class PSOSwarm:
    """
    A vectorized implementation of Particle Swarm Optimization using NumPy arrays.
    Modified to pass previous positions and control parameters for detailed metric calculation.
    Boundary handling updated to assign infinite fitness to infeasible particles.

    Attributes:
        # ... (previous attributes) ...
        previous_positions (np.ndarray): Positions from the previous step.
    """

    def __init__(self,
                 objective_function,
                 num_particles: int,
                 strategy=None,
                 v_clamp_ratio: float = 0.2,
                 use_velocity_clamping: bool = True,
                 convergence_patience: int = 50,
                 convergence_threshold_gbest: float = 1e-8,
                 convergence_threshold_pbest_std: float = 1e-6,
                 stability_threshold: float = 1e-3  # Note: stability_threshold is now used within metrics class if needed
                 ):
        """
        Initializes the vectorized PSO algorithm.
        """
        self.objective_function = objective_function
        self.num_particles = num_particles
        self.dim = objective_function.dim
        self.bounds = objective_function.bounds
        self.kb_sharing_strat = strategy

        # --- Initialize Swarm State ---
        # Initialize positions safely within bounds
        self.positions: np.ndarray = np.random.uniform(low=self.bounds[0], high=self.bounds[1], size=(self.num_particles, self.dim))
        # Ensure initial positions are strictly within bounds if bounds are identical (e.g., for fixed dimension)
        if self.bounds[0] == self.bounds[1]:
            self.positions = np.full((self.num_particles, self.dim), self.bounds[0])
        else:
            # Add small epsilon for strict inequality if needed, or rely on uniform distribution properties
             pass # np.random.uniform is generally sufficient


        self.velocities: np.ndarray = np.zeros((self.num_particles, self.dim))
        self.previous_positions: np.ndarray = self.positions.copy()  # Initialize previous positions

        # Initialize personal bests - evaluate only initial valid positions
        self.pbest_positions: np.ndarray = self.positions.copy()
        self.pbest_values: np.ndarray  # Type annotation for pbest_values
        try:
            if hasattr(self.objective_function, 'evaluate_matrix') and callable(
                    self.objective_function.evaluate_matrix):
                # Evaluate initial positions (which are guaranteed to be within bounds)
                self.pbest_values = self.objective_function.evaluate_matrix(self.positions)  # type: ignore
                log_debug("Initialized pbest using evaluate_matrix.", module_name)
            else:
                log_debug("evaluate_matrix not found, initializing pbest using loop.", module_name)
                self.pbest_values = np.array([self.objective_function.evaluate(p) for p in self.positions])  # type: ignore
        except Exception as e:
            log_error(f"Error during initial pbest evaluation: {e}. Initializing pbest values to infinity.",
                      module_name)
            log_error(traceback.format_exc(), module_name)
            self.pbest_values = np.full(self.num_particles, np.inf, dtype=float)  # type: ignore

        if not np.all(np.isfinite(self.pbest_values)):  # type: ignore
            log_warning("Non-finite initial pbest values detected. Replacing with inf.", module_name)
            self.pbest_values[~np.isfinite(self.pbest_values)] = np.inf  # type: ignore

        # Initialize global best
        min_idx = np.argmin(self.pbest_values)
        if len(self.pbest_values) > 0 and np.isfinite(self.pbest_values[min_idx]):
            self.gbest_position: np.ndarray = self.pbest_positions[min_idx].copy()
            self.gbest_value: float = float(self.pbest_values[min_idx])
        else:
            # Fallback if no finite pbest values found initially
            self.gbest_position = self.positions[0].copy() if self.num_particles > 0 else np.zeros(self.dim)
            self.gbest_value = np.inf
            log_warning("Could not determine initial gbest from pbest values. Initializing gbest to inf.", module_name)

        # --- Velocity Clamping Setup ---
        self.use_velocity_clamping = use_velocity_clamping
        range_width = self.bounds[1] - self.bounds[0]
        # Handle zero range width case for v_max
        self.v_max = v_clamp_ratio * range_width if range_width > 0 else v_clamp_ratio # Use default ratio if range is zero
        log_info(f"Velocity clamping {'enabled' if use_velocity_clamping else 'disabled'}. Vmax: {self.v_max:.4f}", module_name)


        # --- Convergence Tracking ---
        self.convergence_patience = convergence_patience
        self.convergence_threshold_gbest = convergence_threshold_gbest
        self.convergence_threshold_pbest_std = convergence_threshold_pbest_std
        self._gbest_history = collections.deque(maxlen=convergence_patience)
        self._stagnation_counter = 0
        self.reset_convergence_tracking() # Initialize history and counter

        # --- Instantiate Metrics Calculator ---
        try:
            # Pass stability_threshold if the metrics class uses it (e.g., for velocity-based stability)
            # If using Poli's stability, it's calculated based on omega, c1, c2 passed in compute.
            self.metrics_calculator = SwarmMetrics()  # No threshold needed if using Poli's
        except NameError:
            log_error("SwarmMetricsVectorized class not found. Metrics calculation disabled.", module_name)
            self.metrics_calculator = None
        except Exception as e:
             log_error(f"Error initializing SwarmMetricsVectorized: {e}", module_name)
             log_error(traceback.format_exc(), module_name)
             self.metrics_calculator = None


        log_info(f"Initialized Vectorized PSO: {num_particles} particles, {self.dim} dimensions.", module_name)
        log_info(f"Initial GBest Value: {self.gbest_value:.4e}", module_name)

    def optimize_step(self, omega: float, c1: float, c2: float, 
                     step: int = 0, function_name: Optional[str] = None, 
                     run_id: int = 0) -> Tuple[Dict[str, Any], float, bool]:
        """
        Performs one step of the PSO optimization. Handles boundary constraints by
        assigning infinite fitness to particles outside the bounds, as per the paper.

        Args:
            omega (float): Inertia weight used for this step's velocity update.
            c1 (float): Cognitive coefficient used for this step.
            c2 (float): Social coefficient used for this step.
            step (int): Current optimization step number.
            function_name (Optional[str]): Name of the function being optimized.
            run_id (int): ID of the current run.

        Returns:
            tuple: (metrics, gbest_value, converged)
                   - metrics (dict): Dictionary of swarm metrics for the step.
                   - gbest_value (float): The global best value after the step.
                   - converged (bool): True if swarm convergence criteria met.
        """
        # Store current positions before update
        self.previous_positions = self.positions.copy()

        # --- Generate Random Numbers ---
        r1 = np.random.rand(self.num_particles, self.dim)
        r2 = np.random.rand(self.num_particles, self.dim)

        # --- Calculate Velocity Components ---
        cognitive_velocity = c1 * r1 * (self.pbest_positions - self.positions)
        social_velocity = c2 * r2 * (self.gbest_position - self.positions)
        inertia_velocity = omega * self.velocities

        # --- Update Velocities ---
        self.velocities = inertia_velocity + cognitive_velocity + social_velocity

        # --- Apply Velocity Clamping ---
        if self.use_velocity_clamping:
            # Ensure v_max is treated as a scalar or array of correct shape if bounds differ per dimension
            v_max_limit = self.v_max # Assuming scalar v_max based on initialization
            self.velocities = np.clip(self.velocities, -v_max_limit, v_max_limit)


        # --- Update Positions ---
        # Calculate new positions, potentially outside bounds
        self.positions += self.velocities

        # --- Evaluate New Positions (Paper's Boundary Handling) ---
        # 1. Identify infeasible particles (outside bounds)
        lower_bound, upper_bound = self.bounds
        # Check if *any* dimension is out of bounds for each particle
        is_out_of_bounds = np.any((self.positions < lower_bound) | (self.positions > upper_bound), axis=1)
        feasible_mask = ~is_out_of_bounds # Mask for particles *inside* bounds

        # Initialize fitness values to infinity
        fitness_values = np.full(self.num_particles, np.inf, dtype=float)

        # 2. Evaluate only feasible particles
        feasible_positions = self.positions[feasible_mask]
        if feasible_positions.shape[0] > 0: # Check if there are any feasible particles
            try:
                if hasattr(self.objective_function, 'evaluate_matrix') and callable(
                        self.objective_function.evaluate_matrix):
                    evaluated_feasible_fitness = self.objective_function.evaluate_matrix(feasible_positions)
                else:
                    log_debug("evaluate_matrix not found, using loop evaluation for feasible particles.", module_name)
                    evaluated_feasible_fitness = np.array([self.objective_function.evaluate(p) for p in feasible_positions])

                # Check for non-finite values returned by the objective function
                if not np.all(np.isfinite(evaluated_feasible_fitness)):  # type: ignore
                     log_warning("Non-finite fitness values detected from objective function evaluation. Replacing with inf.", module_name)
                     evaluated_feasible_fitness[~np.isfinite(evaluated_feasible_fitness)] = np.inf  # type: ignore

                # Assign evaluated fitness to the correct indices in the full fitness array
                fitness_values[feasible_mask] = evaluated_feasible_fitness

            except Exception as e:
                log_error(f"Error during feasible particle evaluation: {e}. Fitness remains infinity.", module_name)
                log_error(traceback.format_exc(), module_name)
                # fitness_values[feasible_mask] will remain np.inf

        # --- Update Personal Bests ---
        # Only update if the new fitness is better (finite fitness always better than inf)
        # And also better than the previous pbest value.
        improvement_mask = fitness_values < self.pbest_values
        self.pbest_positions[improvement_mask] = self.positions[improvement_mask] # Store the potentially out-of-bounds position if it led to a finite fitness somehow (shouldn't happen with this logic) or if fitness improved
        self.pbest_values[improvement_mask] = fitness_values[improvement_mask]

        # --- Update Global Best ---
        # Find the index of the minimum pbest value (argmin ignores inf)
        if np.all(np.isinf(self.pbest_values)):
             # Handle case where all particles might be infeasible or have inf fitness
             log_warning("All pbest values are infinite. Cannot update gbest.", module_name)
             # gbest_value and gbest_position remain unchanged
        else:
            current_min_idx = np.argmin(self.pbest_values)
            current_gbest_value = self.pbest_values[current_min_idx] # This will be finite if not all are inf

            # Update gbest only if the new best pbest is better than the current gbest
            if current_gbest_value < self.gbest_value:
                self.gbest_value = current_gbest_value
                self.gbest_position = self.pbest_positions[current_min_idx].copy()


        # --- Track GBest History ---
        # Use the current gbest_value (which might be inf if optimization hasn't found a finite value yet)
        value_to_add = self.gbest_value
        # If history is full and the new value is inf, but the history contains finite values,
        # maybe keep the last finite value? For now, append the current gbest value directly.
        self._gbest_history.append(value_to_add)


        # --- Calculate Metrics using External Calculator ---
        # Metrics should be calculated based on the actual positions, even if out of bounds
        metrics = {}
        if self.metrics_calculator:
            try:
                # Pass previous positions and current control parameters
                metrics = self.metrics_calculator.compute(
                    positions=self.positions, # Use current (potentially OOB) positions
                    previous_positions=self.previous_positions,
                    velocities=self.velocities,
                    bounds=self.bounds,
                    omega=omega,
                    c1=c1,
                    c2=c2,
                    step=step,
                    function_name=function_name,
                    run_id=run_id
                )
            except Exception as e:
                log_error(f"Error computing metrics: {e}", module_name)
                log_error(traceback.format_exc(), module_name)
        metrics['gbest_value'] = self.gbest_value  # Add gbest for convenience

        # --- Check for Swarm Convergence ---
        converged = self._check_convergence()

        return metrics, self.gbest_value, converged

    def _check_convergence(self) -> bool:
        """Checks if the swarm has converged based on diversity and gbest stagnation."""
        # 1. Check GBest Stagnation
        gbest_stagnated = False
        if len(self._gbest_history) == self.convergence_patience:
            # Filter out inf values before calculating improvement
            finite_history = [v for v in self._gbest_history if np.isfinite(v)]
            if len(finite_history) > 0 and np.isfinite(self.gbest_value):
                 history_start_val = finite_history[0] # Compare against the oldest finite value
                 improvement = history_start_val - self.gbest_value
                 # Check if improvement is non-negative but below threshold
                 if self.convergence_threshold_gbest > improvement >= 0:
                     self._stagnation_counter += 1
                 else:
                     self._stagnation_counter = 0 # Reset if improvement is significant or negative
            else:
                 # Reset if current gbest is inf or no finite history exists
                 self._stagnation_counter = 0
                 log_debug("Non-finite gbest value or history, resetting stagnation counter.", module_name)

            if self._stagnation_counter >= self.convergence_patience:
                gbest_stagnated = True
                log_debug(f"GBest stagnation detected ({self._stagnation_counter} steps).", module_name)
        else:
             # Not enough history yet, reset counter
            self._stagnation_counter = 0

        # 2. Check Swarm Diversity (using std dev of pbest values)
        finite_pbest_values = self.pbest_values[np.isfinite(self.pbest_values)]
        diversity_low = False
        pbest_std_dev = np.nan

        if len(finite_pbest_values) < 2:
            # If fewer than 2 particles have finite pbest, consider diversity low only if gbest also stagnated
            diversity_low = gbest_stagnated
            log_debug(
                f"Diversity check: Not enough finite pbest values ({len(finite_pbest_values)}). diversity_low set to {diversity_low}.",
                module_name)
        else:
            pbest_std_dev = np.std(finite_pbest_values)
            diversity_low = pbest_std_dev < self.convergence_threshold_pbest_std
            log_debug(
                f"Diversity check: PBest Std Dev = {pbest_std_dev:.2e}. Threshold = {self.convergence_threshold_pbest_std:.2e}. diversity_low = {diversity_low}",
                module_name)

        # 3. Combine Conditions
        if gbest_stagnated and diversity_low:
            log_debug(
                f"Convergence detected: GBest Stagnated ({self._stagnation_counter} steps) & PBest Std Dev ({pbest_std_dev:.2e}) low.",
                module_name)
            return True

        return False

    def reset_convergence_tracking(self):
        """Resets gbest history and stagnation counter."""
        self._gbest_history.clear()
        # Initialize with current gbest, even if it's inf
        initial_gbest = self.gbest_value if hasattr(self, 'gbest_value') else float('inf')
        self._gbest_history.append(initial_gbest)
        self._stagnation_counter = 0
        log_debug("Convergence tracking reset.", module_name)

