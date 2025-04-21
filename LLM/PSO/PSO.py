# File: PSO-ToyBox/LLM/PSO/PSO.py
import numpy as np
import collections # Needed for deque

# Make sure necessary imports are present
from LLM.PSO.Cognitive.PositionSharing import KnowledgeSharingStrategy
from LLM.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction
from LLM.PSO.Particle import Particle
from LLM.PSO.Metrics.SwarmMetrics import SwarmMetrics # Assuming SwarmMetrics is in this path

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
        self.objective_function = objective_function
        self.kb_sharing_strat = strategy
        self.dim = objective_function.dim
        self.bounds = objective_function.bounds
        self.particles = [
            Particle(self.dim, self.bounds) # Assumes Particle class is defined elsewhere
            for _ in range(objective_function.num_particles)
        ]

        self.gbest_position = self.particles[0].position.copy()
        self.gbest_value = float('inf')

        self.use_velocity_clamping = use_velocity_clamping
        self.v_max = v_clamp_ratio * (self.bounds[1] - self.bounds[0])

        self.metrics_calculator = SwarmMetrics()

        # --- Convergence Tracking ---
        self.convergence_patience = convergence_patience
        self.convergence_threshold_gbest = convergence_threshold_gbest
        self.convergence_threshold_pbest_std = convergence_threshold_pbest_std
        self._gbest_history = collections.deque(maxlen=convergence_patience) # Store recent gbest values
        self._stagnation_counter = 0 # Count steps gbest hasn't improved enough

    def _evaluate(self, particle):
        """Evaluates a particle's position and updates its pbest."""
        fitness = self.objective_function.evaluate(particle.position)
        if fitness < particle.pbest_value:
            particle.pbest_value = fitness
            particle.pbest_position = particle.position.copy()
        return fitness

    def _update_velocity(self, particle, omega, c1, c2):
        """Updates a particle's velocity."""
        r1 = np.random.rand(particle.dim)
        r2 = np.random.rand(particle.dim)
        cognitive = c1 * r1 * (particle.pbest_position - particle.position)
        # Get the best position based on the strategy (lbest, gbest, etc.)
        social_target = self.kb_sharing_strat.get_best_position(particle, self.particles)
        social = c2 * r2 * (social_target - particle.position)
        particle.velocity = omega * particle.velocity + cognitive + social

        # Apply velocity clamping if enabled
        if self.use_velocity_clamping:
            particle.velocity = np.clip(particle.velocity, -self.v_max, self.v_max)

    def _update_position(self, particle):
        """Updates a particle's position and ensures it stays within bounds."""
        particle.position += particle.velocity
        particle.position = np.clip(particle.position, self.bounds[0], self.bounds[1])

    def _check_convergence(self) -> bool:
        """
        Checks if the swarm has converged based on diversity and gbest stagnation.

        Returns:
            bool: True if the swarm is considered converged, False otherwise.
        """
        # 1. Check GBest Stagnation
        gbest_stagnated = False
        if len(self._gbest_history) == self.convergence_patience:
            # Calculate improvement over the patience window
            improvement = self._gbest_history[0] - self.gbest_value # Improvement is positive if value decreased
            if improvement < self.convergence_threshold_gbest:
                self._stagnation_counter += 1
            else:
                self._stagnation_counter = 0 # Reset if significant improvement occurred

            # Check if stagnation counter reached patience
            if self._stagnation_counter >= self.convergence_patience:
                 gbest_stagnated = True
        else:
             # Not enough history yet, reset stagnation counter
             self._stagnation_counter = 0


        # 2. Check Swarm Diversity (using std dev of pbest values)
        pbest_values = np.array([p.pbest_value for p in self.particles if np.isfinite(p.pbest_value)])
        if len(pbest_values) < 2: # Cannot calculate std dev with < 2 points
            # If only one particle or all others have inf pbest, consider it non-diverse (converged)
            # Or handle as needed - here we assume it's not converged yet unless gbest also stagnated severely
             diversity_low = False
        else:
            pbest_std_dev = np.std(pbest_values)
            diversity_low = pbest_std_dev < self.convergence_threshold_pbest_std

        # 3. Combine Conditions
        # Converged if gbest has stagnated AND diversity is low
        if gbest_stagnated and diversity_low:
            print(f"Convergence detected: GBest Stagnated ({self._stagnation_counter} steps) & PBest Std Dev ({pbest_std_dev:.2e}) low.")
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

            # Update global best if this particle is better
            if fitness < self.gbest_value:
                self.gbest_value = fitness
                self.gbest_position = particle.position.copy()

        # --- Track GBest History for Stagnation Check ---
        self._gbest_history.append(self.gbest_value)

        # --- Calculate Metrics ---
        # Use all particles for standard metrics calculation
        metrics = self.metrics_calculator.compute(self.particles, self.bounds)

        # --- Check for Swarm Convergence ---
        converged = self._check_convergence()

        return metrics, self.gbest_value, converged

    def reset_convergence_tracking(self):
        """Resets gbest history and stagnation counter, e.g., at the start of an episode."""
        self._gbest_history.clear()
        self._stagnation_counter = 0
        # Optionally re-initialize gbest value if needed for a full reset
        # self.gbest_value = float('inf')
        # self.gbest_position = self.particles[0].position.copy() # Or re-randomize

