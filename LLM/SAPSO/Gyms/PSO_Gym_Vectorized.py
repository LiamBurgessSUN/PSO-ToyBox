# File: PSO-ToyBox/LLM/SAPSO/Gyms/PSOEnvVectorized.py
# --- Imports ---
import gymnasium as gym # Use gymnasium instead of gym
import numpy as np
import math
import collections

# Import the new vectorized PSO implementation
from LLM.PSO.PsoVectorized import PSOVectorized

# Example function (can be changed)
from LLM.PSO.ObjectiveFunctions.Training.Rastrgin import RastriginFunction
# Strategy import kept for passing to PSO, but GBest is assumed by PSOVectorized
from LLM.PSO.Cognitive.LBest import LocalBestStrategy


class PSOEnvVectorized(gym.Env):
    """
    Gym environment for PSO using the vectorized PSO implementation (PSOVectorized).
    Adapts PSO control parameters based on RL agent actions.
    """
    metadata = {'render_modes': [], 'render_fps': 4}

    def __init__(self,
                 obj_func=RastriginFunction(dim=30), # Removed num_particles here
                 num_particles=30, # Added num_particles parameter
                 max_steps=5000,
                 agent_step_size=10,
                 adaptive_nt=False,
                 nt_range=(1, 100),
                 v_clamp_ratio=0.2, # Added PSO param
                 use_velocity_clamping=True, # Added PSO param
                 convergence_patience=50,
                 convergence_threshold_gbest=1e-8,
                 convergence_threshold_pbest_std=1e-6,
                 stability_threshold=1e-3 # Added metrics param
                 ):
        """
        Initializes the Vectorized PSO Environment.

        Args:
            obj_func: An instance of an objective function class (must have evaluate, optionally evaluate_matrix).
            num_particles (int): Number of particles in the swarm.
            max_steps (int): Maximum number of PSO steps allowed per episode.
            agent_step_size (int): Fixed number of PSO steps if adaptive_nt is False.
            adaptive_nt (bool): If True, agent learns the number of steps (nt).
            nt_range (tuple): (min, max) allowed values for nt if adaptive_nt is True.
            v_clamp_ratio (float): PSO velocity clamping ratio.
            use_velocity_clamping (bool): PSO flag to enable velocity clamping.
            convergence_patience (int): Steps gbest must stagnate for convergence check.
            convergence_threshold_gbest (float): Max gbest improvement for stagnation.
            convergence_threshold_pbest_std (float): Max pbest std dev for convergence.
            stability_threshold (float): Threshold for stability metric calculation.
        """
        super().__init__()
        self.max_steps = max_steps
        self.current_step = 0
        self.last_gbest = float('inf')
        self.adaptive_nt = adaptive_nt
        self.nt_range = nt_range
        self._current_nt = agent_step_size if not self.adaptive_nt else self.nt_range[0]

        # Store parameters needed for PSOVectorized initialization
        self.obj_fn = obj_func
        self.num_particles = num_particles
        self.v_clamp_ratio = v_clamp_ratio
        self.use_velocity_clamping = use_velocity_clamping
        self.convergence_patience = convergence_patience
        self.convergence_threshold_gbest = convergence_threshold_gbest
        self.convergence_threshold_pbest_std = convergence_threshold_pbest_std
        self.stability_threshold = stability_threshold

        # --- Initialize PSOVectorized ---
        # Strategy object is passed but PSOVectorized currently assumes G-Best
        self.strategy = LocalBestStrategy(neighborhood_size=2) # Example strategy instance
        self.pso = PSOVectorized(
            objective_function=self.obj_fn,
            num_particles=self.num_particles,
            strategy=self.strategy, # Pass strategy object
            v_clamp_ratio=self.v_clamp_ratio,
            use_velocity_clamping=self.use_velocity_clamping,
            convergence_patience=self.convergence_patience,
            convergence_threshold_gbest=self.convergence_threshold_gbest,
            convergence_threshold_pbest_std=self.convergence_threshold_pbest_std,
            stability_threshold=self.stability_threshold
        )
        self.last_gbest = self.pso.gbest_value # Initialize last_gbest after PSO init

        # === Observation Space ===
        # [squashed_norm_avg_vel, feasible_ratio, stability_ratio, percent_completion]
        # Note: Keys used in _get_obs must match those from SwarmMetricsVectorized
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            shape=(4,),
            dtype=np.float32
        )

        # === Action Space ===
        self.action_bounds_low = np.array([0.3, 0.0, 0.0], dtype=np.float32) # omega, c1, c2
        self.action_bounds_high = np.array([1.0, 3.0, 3.0], dtype=np.float32)
        self.action_dim = 3

        if self.adaptive_nt:
            self.action_bounds_low = np.append(self.action_bounds_low, -1.0).astype(np.float32) # nt action range [-1, 1]
            self.action_bounds_high = np.append(self.action_bounds_high, 1.0).astype(np.float32)
            self.action_dim = 4

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

    def _rescale_action(self, action: np.ndarray) -> tuple:
        """ Rescales agent action [-1, 1] to actual parameter bounds. """
        # Rescale omega, c1, c2
        low_cp = self.action_bounds_low[:3]
        high_cp = self.action_bounds_high[:3]
        rescaled_cp = low_cp + (action[:3] + 1.0) * 0.5 * (high_cp - low_cp)
        rescaled_cp = np.clip(rescaled_cp, low_cp, high_cp)
        omega, c1, c2 = rescaled_cp

        # Determine nt value
        nt = self._current_nt # Default for fixed mode
        if self.adaptive_nt:
            action_nt = action[3]
            low_nt, high_nt = self.nt_range
            rescaled_nt = low_nt + (action_nt + 1.0) * 0.5 * (high_nt - low_nt)
            nt = int(np.round(np.clip(rescaled_nt, low_nt, high_nt)))

        return omega, c1, c2, nt

    def reset(self, seed=None, options=None):
        """ Resets the environment for a new episode. """
        super().reset(seed=seed)
        self.current_step = 0
        self._current_nt = self.nt_range[0] if self.adaptive_nt else self._current_nt

        # Re-initialize PSOVectorized state
        self.pso = PSOVectorized(
            objective_function=self.obj_fn,
            num_particles=self.num_particles,
            strategy=self.strategy, # Pass strategy object
            v_clamp_ratio=self.v_clamp_ratio,
            use_velocity_clamping=self.use_velocity_clamping,
            convergence_patience=self.convergence_patience,
            convergence_threshold_gbest=self.convergence_threshold_gbest,
            convergence_threshold_pbest_std=self.convergence_threshold_pbest_std,
            stability_threshold=self.stability_threshold
        )
        self.last_gbest = self.pso.gbest_value # Reset last_gbest

        # Get initial metrics from the PSO instance's calculator
        initial_metrics = self.pso.metrics_calculator.compute(
            self.pso.positions, self.pso.velocities, self.pso.bounds
        )

        observation = self._get_obs(initial_metrics)
        info = {'initial_gbest': self.pso.gbest_value} # Add initial gbest to info
        return observation, info

    def step(self, action: np.ndarray):
        """ Executes one agent step (running `nt` internal PSO steps). """
        # 1. Rescale action
        omega, c1, c2, next_nt = self._rescale_action(action)

        # 2. Update next nt if adaptive
        if self.adaptive_nt:
             self._current_nt = next_nt

        # 3. Initialize variables for this agent turn
        cumulative_reward = 0.0
        terminated = False
        truncated = False
        steps_taken_this_turn = 0
        step_metrics_list = [] # To store metrics from each internal PSO step

        current_omega, current_c1, current_c2 = omega, c1, c2

        # 4. Run internal PSO loop for `self._current_nt` steps
        for _ in range(self._current_nt):
            if self.current_step >= self.max_steps:
                truncated = True
                # print(f"Episode truncated at step {self.current_step} (max_steps reached).") # Optional debug
                break

            # --- Execute one PSO Step ---
            step_metrics, current_gbest, converged_this_step = self.pso.optimize_step(
                current_omega, current_c1, current_c2
            )
            self.current_step += 1
            steps_taken_this_turn += 1

            # --- Calculate Reward ---
            step_reward = self._calculate_relative_reward(current_gbest)
            self.last_gbest = current_gbest
            cumulative_reward += step_reward

            # --- Store Detailed Metrics for Info Dict ---
            # Add CPs used and step index to the metrics dict from PSO
            detailed_metrics_for_info = step_metrics.copy()
            detailed_metrics_for_info['omega'] = current_omega
            detailed_metrics_for_info['c1'] = current_c1
            detailed_metrics_for_info['c2'] = current_c2
            detailed_metrics_for_info['pso_step'] = self.current_step - 1
            step_metrics_list.append(detailed_metrics_for_info)

            # --- Check Termination Conditions ---
            if converged_this_step:
                terminated = True
                # print(f"PSO Swarm converged at step {self.current_step}.") # Optional debug
                break

            if self.current_step >= self.max_steps: # Check again after step
                truncated = True
                # print(f"Episode truncated at step {self.current_step} (max_steps reached).") # Optional debug
                break

        # 5. Prepare Observation for the *next* agent step
        # Calculate metrics based on the final state after the internal loop
        final_metrics_for_obs = self.pso.metrics_calculator.compute(
            self.pso.positions, self.pso.velocities, self.pso.bounds
        )
        observation = self._get_obs(final_metrics_for_obs)

        # 6. Prepare Info Dictionary
        info = {
            'steps_taken': steps_taken_this_turn,
            'nt': self._current_nt,
            'step_metrics': step_metrics_list, # List of metrics from each internal step
            'final_gbest': self.pso.gbest_value # Gbest at the end of the agent turn
        }

        terminated = bool(terminated)
        truncated = bool(truncated)

        return observation, cumulative_reward, terminated, truncated, info

    def _calculate_relative_reward(self, current_gbest: float) -> float:
        """ Calculates reward based on relative improvement in gbest. """
        y_old = self.last_gbest
        y_new = current_gbest

        # Handle non-finite cases or no change
        if not np.isfinite(y_new) or np.isclose(y_new, y_old):
            return 0.0
        if not np.isfinite(y_old) and np.isfinite(y_new): # Found first finite solution
             # Assign a positive reward, magnitude could be tuned
             # Using 1.0 as in the paper's description for y_old > 0, y_new < 0 case
             return 1.0

        # Apply relative reward formula (Equation 25 from paper)
        reward = 0.0
        beta = np.abs(y_old) + np.abs(y_new)

        if y_new >= 0 and y_old >= 0: # Both non-negative
            # Shift to ensure positivity for division
            shifted_old = y_old + beta
            shifted_new = y_new + beta
            if not np.isclose(shifted_old, 0):
                # Improvement means y_new < y_old => shifted_new < shifted_old
                reward = 2.0 * (shifted_old - shifted_new) / shifted_old
        elif y_new < 0 and y_old < 0: # Both negative
            # Shift to ensure positivity for division
            shift = 2 * beta
            shifted_old = y_old + shift
            shifted_new = y_new + shift
            if not np.isclose(shifted_old, 0):
                # Improvement means y_new < y_old => shifted_new < shifted_old
                reward = 2.0 * (shifted_old - shifted_new) / shifted_old
        elif y_new < 0 and y_old >= 0: # Improvement crossed zero
            # Assign max reward as per paper's description
            reward = 1.0
        # Case y_new >= 0 and y_old < 0 (Worsened across zero) is implicitly 0.0

        # Ensure reward is non-negative and capped (though formula should handle this)
        return np.clip(reward, 0.0, 1.0)


    def _get_obs(self, metrics: dict) -> np.ndarray:
         """ Calculates the observation vector from swarm metrics. """
         # Use keys returned by SwarmMetricsVectorized
         avg_vel = metrics.get('avg_velocity_magnitude', 0.0)
         feasible_ratio = metrics.get('feasible_ratio', 0.0)
         # Use 'stability_ratio' key from SwarmMetricsVectorized
         stable_ratio = metrics.get('stability_ratio', 0.0)

         # Normalize average velocity using tanh squashing (Eq 22)
         l, u = self.pso.bounds
         range_width = u - l
         if np.isclose(range_width, 0):
              normalized_vel_component = 0.0
         else:
             # Paper uses v - (l+u)/2, but avg_vel is magnitude (>=0).
             # Let's adapt: compare avg_vel to half the range width as a scale.
             # Normalize velocity relative to the typical scale of movement.
             scale = 0.5 * range_width
             normalized_vel_component = math.tanh(avg_vel / scale if scale > 0 else avg_vel)

         # Squash to [0, 1]
         squashed_norm_avg_vel = (normalized_vel_component + 1.0) / 2.0

         # Calculate percentage completion
         percent_completion = self.current_step / self.max_steps if self.max_steps > 0 else 1.0

         # Construct the observation vector
         obs_vec = np.array([
             np.clip(squashed_norm_avg_vel, 0.0, 1.0),
             np.clip(feasible_ratio, 0.0, 1.0),
             np.clip(stable_ratio, 0.0, 1.0),
             np.clip(percent_completion, 0.0, 1.0)
         ], dtype=np.float32)

         # Sanity check for NaN or Inf values
         if not np.all(np.isfinite(obs_vec)):
             print(f"Warning: Non-finite values detected in observation at step {self.current_step}. Clipping/Replacing.")
             obs_vec = np.nan_to_num(obs_vec, nan=0.0, posinf=1.0, neginf=0.0)
             obs_vec = np.clip(obs_vec, 0.0, 1.0)

         return obs_vec

    # --- Standard Gym methods (optional) ---
    def render(self, mode='human'):
        pass # Implement visualization if needed

    def close(self):
        pass # Clean up resources if needed

