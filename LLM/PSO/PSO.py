# File: PSO-ToyBox/LLM/PSO/PSO.py
# Refactored to use the logger module
import traceback

import numpy as np
import collections # Needed for deque
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
# Make sure necessary imports are present
from LLM.PSO.Cognitive.PositionSharing import KnowledgeSharingStrategy
from LLM.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction
from LLM.PSO.Particle import Particle
from LLM.PSO.Metrics.SwarmMetrics import SwarmMetrics # Assuming SwarmMetrics is in this path

# --- Module Name for Logging ---
# It's often better to define this once at the module level
module_name = Path(__file__).stem # Gets 'PSO'

class PSO:
    def __init__(self, objective_function: ObjectiveFunction,
                 strategy: KnowledgeSharingStrategy,
                 v_clamp_ratio=0.1,
                 use_velocity_clamping=True,
                 # --- Add Convergence Parameters ---
                 convergence_patience=50,          # Steps gbest must stagnate
                 convergence_threshold_gbest=1e-8, # Max gbest improvement to be considered stagnant
                 convergence_threshold_pbest_std=1e-6 # Max std dev of pbest values for convergence
                 ):
        """
        Initializes the standard PSO algorithm.

        Args:
            objective_function (ObjectiveFunction): The function to minimize.
            strategy (KnowledgeSharingStrategy): The strategy for social interaction (e.g., GBest, LBest).
            v_clamp_ratio (float): Ratio of search space range for velocity clamping.
            use_velocity_clamping (bool): Whether to enable velocity clamping.
            convergence_patience (int): Number of steps gbest must stagnate for convergence.
            convergence_threshold_gbest (float): Max gbest improvement considered stagnation.
            convergence_threshold_pbest_std (float): Max pbest std dev for convergence.
        """
        self.objective_function = objective_function
        self.kb_sharing_strat = strategy
        self.dim = objective_function.dim
        self.bounds = objective_function.bounds
        self.num_particles = objective_function.num_particles # Store num_particles
        self.particles = [
            Particle(self.dim, self.bounds)
            for _ in range(self.num_particles) # Use stored num_particles
        ]

        self.gbest_position = self.particles[0].position.copy() if self.particles else np.zeros(self.dim)
        self.gbest_value = float('inf')

        self.use_velocity_clamping = use_velocity_clamping
        # Calculate v_max safely, handle potential zero range
        range_width = self.bounds[1] - self.bounds[0]
        self.v_max = v_clamp_ratio * range_width if range_width > 0 else v_clamp_ratio

        self.metrics_calculator = SwarmMetrics()

        # --- Convergence Tracking ---
        self.convergence_patience = convergence_patience
        self.convergence_threshold_gbest = convergence_threshold_gbest
        self.convergence_threshold_pbest_std = convergence_threshold_pbest_std
        # Initialize deque with maxlen
        self._gbest_history = collections.deque(maxlen=convergence_patience)
        self._stagnation_counter = 0

        # Initialize particles and gbest properly
        self._initialize_swarm()

    def _initialize_swarm(self):
        """Initializes particle pbest values and finds initial gbest."""
        if not self.particles:
            log_warning("Cannot initialize swarm: No particles created.", module_name)
            return

        initial_gbest = float('inf')
        initial_gbest_pos = self.particles[0].position.copy()

        for p in self.particles:
            try:
                fitness = self.objective_function.evaluate(p.position)
                # Handle non-finite fitness values during initialization
                if not np.isfinite(fitness):
                    log_warning(f"Non-finite fitness ({fitness}) for particle at {p.position}. Setting pbest to inf.", module_name)
                    p.pbest_value = float('inf')
                    # Keep p.pbest_position as the initial random position
                else:
                    p.pbest_value = fitness
                    p.pbest_position = p.position.copy()
                    if fitness < initial_gbest:
                        initial_gbest = fitness
                        initial_gbest_pos = p.position.copy()
            except Exception as e:
                log_error(f"Error evaluating initial position for particle: {e}", module_name)
                log_error(traceback.format_exc(), module_name)
                p.pbest_value = float('inf') # Set pbest to inf on error

        self.gbest_value = initial_gbest
        self.gbest_position = initial_gbest_pos
        # Reset convergence tracking after initialization
        self.reset_convergence_tracking()
        log_info(f"Swarm initialized. Initial GBest: {self.gbest_value:.6e}", module_name)


    def _evaluate(self, particle):
        """Evaluates a particle's position and updates its pbest."""
        try:
            fitness = self.objective_function.evaluate(particle.position)
            # Handle non-finite fitness values during optimization steps
            if not np.isfinite(fitness):
                 log_debug(f"Non-finite fitness ({fitness}) encountered during evaluation for particle. Ignoring for pbest update.", module_name)
                 return float('inf') # Return inf so it doesn't update pbest incorrectly

            if fitness < particle.pbest_value:
                particle.pbest_value = fitness
                particle.pbest_position = particle.position.copy()
            return fitness
        except Exception as e:
            log_error(f"Error evaluating particle position {particle.position}: {e}", module_name)
            # Potentially log traceback if needed: log_error(traceback.format_exc(), module_name)
            return float('inf') # Return inf on error

    def _update_velocity(self, particle, omega, c1, c2):
        """Updates a particle's velocity."""
        r1 = np.random.rand(particle.dim)
        r2 = np.random.rand(particle.dim)
        cognitive = c1 * r1 * (particle.pbest_position - particle.position)

        # Get the best position based on the strategy (lbest, gbest, etc.)
        # Ensure strategy object is correctly initialized and linked if needed
        if self.kb_sharing_strat:
            try:
                social_target = self.kb_sharing_strat.get_best_position(particle, self.particles)
            except Exception as e:
                log_error(f"Error getting social target from strategy: {e}. Using gbest as fallback.", module_name)
                social_target = self.gbest_position # Fallback to gbest
        else:
            log_warning("Knowledge sharing strategy not set. Using gbest as default social target.", module_name)
            social_target = self.gbest_position # Default to gbest if no strategy

        social = c2 * r2 * (social_target - particle.position)
        particle.velocity = omega * particle.velocity + cognitive + social

        # Apply velocity clamping if enabled
        if self.use_velocity_clamping:
            # Ensure v_max is a scalar or has the same dim as velocity
            if isinstance(self.v_max, (int, float)):
                particle.velocity = np.clip(particle.velocity, -self.v_max, self.v_max)
            elif isinstance(self.v_max, np.ndarray) and self.v_max.shape == particle.velocity.shape:
                 particle.velocity = np.clip(particle.velocity, -self.v_max, self.v_max)
            else:
                 log_warning(f"v_max type ({type(self.v_max)}) or shape mismatch. Skipping clamping.", module_name)


    def _update_position(self, particle):
        """Updates a particle's position and ensures it stays within bounds."""
        particle.position += particle.velocity
        # Ensure bounds are handled correctly (clip expects scalar or array bounds)
        low_bound, high_bound = self.bounds
        particle.position = np.clip(particle.position, low_bound, high_bound)

    def _check_convergence(self) -> bool:
        """
        Checks if the swarm has converged based on diversity and gbest stagnation.

        Returns:
            bool: True if the swarm is considered converged, False otherwise.
        """
        # 1. Check GBest Stagnation
        gbest_stagnated = False
        if len(self._gbest_history) == self.convergence_patience:
            # Ensure history contains finite values before comparing
            history_start_val = self._gbest_history[0]
            if np.isfinite(history_start_val) and np.isfinite(self.gbest_value):
                 improvement = history_start_val - self.gbest_value # Improvement is positive if value decreased
                 # Check if improvement is less than threshold (allow for floating point issues)
                 if improvement < self.convergence_threshold_gbest: # improvement >= 0
                     self._stagnation_counter += 1
                 else:
                     self._stagnation_counter = 0 # Reset if significant improvement
            else:
                 # If values are non-finite, reset stagnation (cannot determine improvement)
                 self._stagnation_counter = 0
                 log_debug("Non-finite gbest value in history, resetting stagnation counter.", module_name)

            # Check if stagnation counter reached patience
            if self._stagnation_counter >= self.convergence_patience:
                 gbest_stagnated = True
                 log_debug(f"GBest stagnation detected ({self._stagnation_counter} steps).", module_name)
        else:
             # Not enough history yet, reset stagnation counter
             self._stagnation_counter = 0


        # 2. Check Swarm Diversity (using std dev of pbest values)
        # Filter out infinite pbest values before calculating std dev
        pbest_values = np.array([p.pbest_value for p in self.particles if np.isfinite(p.pbest_value)])
        diversity_low = False
        pbest_std_dev = np.nan # Default if calculation fails

        if len(pbest_values) < 2: # Cannot calculate std dev with < 2 finite points
            # If only one particle or all others have inf pbest, consider it non-diverse only if gbest also stagnated
            diversity_low = gbest_stagnated
            log_debug(f"Diversity check: Not enough finite pbest values ({len(pbest_values)}). diversity_low set based on gbest_stagnated ({diversity_low}).", module_name)
        else:
            pbest_std_dev = np.std(pbest_values)
            diversity_low = pbest_std_dev < self.convergence_threshold_pbest_std
            log_debug(f"Diversity check: PBest Std Dev = {pbest_std_dev:.2e}. Threshold = {self.convergence_threshold_pbest_std:.2e}. diversity_low = {diversity_low}", module_name)

        # 3. Combine Conditions
        # Converged if gbest has stagnated AND diversity is low
        if gbest_stagnated and diversity_low:
            # Use log_debug instead of print for convergence message
            log_debug(f"Convergence detected: GBest Stagnated ({self._stagnation_counter} steps) & PBest Std Dev ({pbest_std_dev:.2e}) low.", module_name)
            return True

        return False

    def optimize_step(self, omega, c1, c2):
        """
        Performs one step of the PSO optimization, updating all particles
        and checking for swarm convergence.

        Args:
            omega (float): Inertia weight.
            c1 (float): Cognitive coefficient.
            c2 (float): Social coefficient.

        Returns:
            tuple: (metrics, gbest_value, converged)
                   - metrics (dict): Dictionary of swarm metrics for the step.
                   - gbest_value (float): The global best value after the step.
                   - converged (bool): True if swarm convergence criteria met, False otherwise.
        """
        current_step_gbest = self.gbest_value # Store gbest before updates

        # Update each particle
        for particle in self.particles:
            # Note: We are not implementing individual particle termination here anymore
            # as per the user's request to terminate the *episode* based on swarm state.
            self._update_velocity(particle, omega, c1, c2)
            self._update_position(particle)
            fitness = self._evaluate(particle) # Updates pbest

            # Update global best if this particle is better and fitness is finite
            if np.isfinite(fitness) and fitness < self.gbest_value:
                self.gbest_value = fitness
                self.gbest_position = particle.position.copy()

        # --- Track GBest History for Stagnation Check ---
        # Ensure only finite values are added to history if possible
        value_to_add = self.gbest_value if np.isfinite(self.gbest_value) else self._gbest_history[-1] if self._gbest_history else float('inf')
        self._gbest_history.append(value_to_add)


        # --- Calculate Metrics ---
        # Use all particles for standard metrics calculation
        try:
            metrics = self.metrics_calculator.compute(self.particles, self.bounds)
        except Exception as e:
            log_error(f"Error computing metrics: {e}", module_name)
            metrics = {} # Return empty dict on error

        # --- Check for Swarm Convergence ---
        converged = self._check_convergence()

        return metrics, self.gbest_value, converged

    def reset_convergence_tracking(self):
        """Resets gbest history and stagnation counter, e.g., at the start of an episode."""
        self._gbest_history.clear()
        self._stagnation_counter = 0
        # Add the current gbest value to start the history
        initial_gbest = self.gbest_value if hasattr(self, 'gbest_value') and np.isfinite(self.gbest_value) else float('inf')
        self._gbest_history.append(initial_gbest)
        log_debug("Convergence tracking reset.", module_name)

