# File: PSO-ToyBox/LLM/PSO/PsoVectorized.py
# Modified to store previous positions and pass necessary parameters
# to the metrics calculator for paper-aligned metric calculations.

import numpy as np
import collections
import traceback  # For logging exceptions
from pathlib import Path  # To get module name

from LLM.Logs.logger import *
from LLM.SAPSO.PSO.Metrics.SwarmMetricsVectorized import SwarmMetricsVectorized

# --- Module Name for Logging ---
module_name = Path(__file__).stem  # Gets 'PsoVectorized'


class PSOVectorized:
    """
    A vectorized implementation of Particle Swarm Optimization using NumPy arrays.
    Modified to pass previous positions and control parameters for detailed metric calculation.

    Attributes:
        # ... (previous attributes) ...
        previous_positions (np.ndarray): Positions from the previous step.
    """

    def __init__(self,
                 objective_function,
                 num_particles,
                 strategy=None,
                 v_clamp_ratio=0.2,
                 use_velocity_clamping=True,
                 convergence_patience=50,
                 convergence_threshold_gbest=1e-8,
                 convergence_threshold_pbest_std=1e-6,
                 stability_threshold=1e-3  # Note: stability_threshold is now used within metrics class if needed
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
        self.positions = np.random.uniform(low=self.bounds[0], high=self.bounds[1], size=(self.num_particles, self.dim))
        self.velocities = np.zeros((self.num_particles, self.dim))
        self.previous_positions = self.positions.copy()  # Initialize previous positions

        # Initialize personal bests
        self.pbest_positions = self.positions.copy()
        try:
            if hasattr(self.objective_function, 'evaluate_matrix') and callable(
                    self.objective_function.evaluate_matrix):
                self.pbest_values = self.objective_function.evaluate_matrix(self.positions)
                log_debug("Initialized pbest using evaluate_matrix.", module_name)
            else:
                log_debug("evaluate_matrix not found, initializing pbest using loop.", module_name)
                self.pbest_values = np.array([self.objective_function.evaluate(p) for p in self.positions])
        except Exception as e:
            log_error(f"Error during initial pbest evaluation: {e}. Initializing pbest values to infinity.",
                      module_name)
            log_error(traceback.format_exc(), module_name)
            self.pbest_values = np.full(self.num_particles, np.inf)

        if not np.all(np.isfinite(self.pbest_values)):
            log_warning("Non-finite initial pbest values detected. Replacing with inf.", module_name)
            self.pbest_values[~np.isfinite(self.pbest_values)] = np.inf

        # Initialize global best
        min_idx = np.argmin(self.pbest_values)
        if len(self.pbest_values) > 0 and np.isfinite(self.pbest_values[min_idx]):
            self.gbest_position = self.pbest_positions[min_idx].copy()
            self.gbest_value = self.pbest_values[min_idx]
        else:
            self.gbest_position = self.positions[0].copy() if self.num_particles > 0 else np.zeros(self.dim)
            self.gbest_value = np.inf
            log_warning("Could not determine initial gbest from pbest values. Initializing gbest to inf.", module_name)

        # --- Velocity Clamping Setup ---
        self.use_velocity_clamping = use_velocity_clamping
        range_width = self.bounds[1] - self.bounds[0]
        self.v_max = v_clamp_ratio * range_width if range_width > 0 else v_clamp_ratio

        # --- Convergence Tracking ---
        self.convergence_patience = convergence_patience
        self.convergence_threshold_gbest = convergence_threshold_gbest
        self.convergence_threshold_pbest_std = convergence_threshold_pbest_std
        self._gbest_history = collections.deque(maxlen=convergence_patience)
        self._stagnation_counter = 0
        self.reset_convergence_tracking()

        # --- Instantiate Metrics Calculator ---
        try:
            # Pass stability_threshold if the metrics class uses it (e.g., for velocity-based stability)
            # If using Poli's stability, it's calculated based on omega, c1, c2 passed in compute.
            self.metrics_calculator = SwarmMetricsVectorized()  # No threshold needed if using Poli's
        except NameError:
            log_error("SwarmMetricsVectorized class not found. Metrics calculation disabled.", module_name)
            self.metrics_calculator = None

        log_info(f"Initialized Vectorized PSO: {num_particles} particles, {self.dim} dimensions.", module_name)
        log_info(f"Initial GBest Value: {self.gbest_value:.4e}", module_name)

    def optimize_step(self, omega, c1, c2):
        """
        Performs one step of the PSO optimization.

        Args:
            omega (float): Inertia weight used for this step's velocity update.
            c1 (float): Cognitive coefficient used for this step.
            c2 (float): Social coefficient used for this step.

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
            self.velocities = np.clip(self.velocities, -self.v_max, self.v_max)

        # --- Update Positions ---
        self.positions += self.velocities

        # --- Apply Boundary Constraints ---
        # TODO This must disable particles instead of clipping
        self.positions = np.clip(self.positions, self.bounds[0], self.bounds[1])

        # --- Evaluate New Positions ---
        fitness_values = np.full(self.num_particles, np.inf)
        try:
            if hasattr(self.objective_function, 'evaluate_matrix') and callable(
                    self.objective_function.evaluate_matrix):
                fitness_values = self.objective_function.evaluate_matrix(self.positions)
                if fitness_values.shape != (self.num_particles,):
                    log_warning(
                        f"evaluate_matrix returned unexpected shape {fitness_values.shape}. Falling back to loop.",
                        module_name)
                    fitness_values = np.array([self.objective_function.evaluate(p) for p in self.positions])
            else:
                log_debug("evaluate_matrix not found, using loop evaluation.", module_name)
                fitness_values = np.array([self.objective_function.evaluate(p) for p in self.positions])
        except Exception as e:
            log_error(f"Error during vectorized evaluation: {e}. Setting fitness to infinity.", module_name)
            log_error(traceback.format_exc(), module_name)

        if not np.all(np.isfinite(fitness_values)):
            log_warning("Non-finite fitness values detected during evaluation. Replacing with inf.", module_name)
            fitness_values[~np.isfinite(fitness_values)] = np.inf

        # --- Update Personal Bests ---
        improvement_mask = fitness_values < self.pbest_values
        self.pbest_positions[improvement_mask] = self.positions[improvement_mask]
        self.pbest_values[improvement_mask] = fitness_values[improvement_mask]

        # --- Update Global Best ---
        current_min_idx = np.argmin(self.pbest_values)
        if np.isfinite(self.pbest_values[current_min_idx]):
            current_gbest_value = self.pbest_values[current_min_idx]
            if current_gbest_value < self.gbest_value:
                self.gbest_value = current_gbest_value
                self.gbest_position = self.pbest_positions[current_min_idx].copy()

        # --- Track GBest History ---
        value_to_add = self.gbest_value if np.isfinite(self.gbest_value) else (
            self._gbest_history[-1] if self._gbest_history else float('inf'))
        self._gbest_history.append(value_to_add)

        # --- Calculate Metrics using External Calculator ---
        metrics = {}
        if self.metrics_calculator:
            try:
                # Pass previous positions and current control parameters
                metrics = self.metrics_calculator.compute(
                    positions=self.positions,
                    previous_positions=self.previous_positions,  # Pass previous positions
                    velocities=self.velocities,
                    bounds=self.bounds,
                    omega=omega,  # Pass omega
                    c1=c1,  # Pass c1
                    c2=c2  # Pass c2
                )
            except Exception as e:
                log_error(f"Error computing metrics: {e}", module_name)
                log_error(traceback.format_exc(), module_name)  # Log traceback for metrics error
        metrics['gbest_value'] = self.gbest_value  # Add gbest for convenience

        # --- Check for Swarm Convergence ---
        converged = self._check_convergence()

        return metrics, self.gbest_value, converged

    def _check_convergence(self) -> bool:
        """Checks if the swarm has converged based on diversity and gbest stagnation."""
        # 1. Check GBest Stagnation
        gbest_stagnated = False
        if len(self._gbest_history) == self.convergence_patience:
            history_start_val = self._gbest_history[0]
            if np.isfinite(history_start_val) and np.isfinite(self.gbest_value):
                improvement = history_start_val - self.gbest_value
                if self.convergence_threshold_gbest > improvement >= 0:
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
        initial_gbest = self.gbest_value if hasattr(self, 'gbest_value') and np.isfinite(self.gbest_value) else float(
            'inf')
        self._gbest_history.append(initial_gbest)
        self._stagnation_counter = 0
        log_debug("Convergence tracking reset.", module_name)
