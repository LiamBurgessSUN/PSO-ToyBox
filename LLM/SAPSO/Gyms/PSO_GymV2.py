# File: PSO-ToyBox/LLM/SAPSO/Gyms/PSO_GymV2.py
# --- Imports ---
import gym
import numpy as np
import math

# Assuming these imports are correctly set up based on your project structure
# Use the modified PSO class that returns the 'converged' flag
from LLM.PSO.PSO import PSO
# Example function and strategy (can be changed)
from LLM.PSO.ObjectiveFunctions.Training.Functions.Rastrgin import RastriginFunction
from LLM.PSO.Cognitive.LBest import LocalBestStrategy
# Assuming SwarmMetrics and compute_swarm_metrics are available
from LLM.PSO.Metrics.SwarmMetrics import SwarmMetrics, compute_swarm_metrics

class PSOEnv(gym.Env):
    """
    Gym environment for PSO adapted to align more closely with the paper.
    Includes modifications to handle early termination based on swarm convergence.
    """
    # Define metadata for Gym compatibility (optional but good practice)
    metadata = {'render_modes': [], 'render_fps': 4}

    def __init__(self,
                 obj_func=RastriginFunction(dim=30, num_particles=30),
                 max_steps=5000,
                 agent_step_size=10, # Renamed from nt for clarity when fixed
                 adaptive_nt=False,  # Flag to toggle adaptive nt
                 nt_range=(1, 100),  # Range for adaptive nt (min_nt, max_nt)
                 # --- Convergence Params for PSO ---
                 # Pass these through to the PSO instance
                 convergence_patience=50,
                 convergence_threshold_gbest=1e-8,
                 convergence_threshold_pbest_std=1e-6
                 ):
        """
        Initializes the PSO Environment.

        Args:
            obj_func: An instance of an objective function class.
            max_steps (int): Maximum number of PSO steps allowed per episode.
            agent_step_size (int): Fixed number of PSO steps if adaptive_nt is False.
            adaptive_nt (bool): If True, agent learns the number of steps (nt).
            nt_range (tuple): (min, max) allowed values for nt if adaptive_nt is True.
            convergence_patience (int): Steps gbest must stagnate for convergence check.
            convergence_threshold_gbest (float): Max gbest improvement for stagnation.
            convergence_threshold_pbest_std (float): Max pbest std dev for convergence.
        """
        super().__init__()
        self.history = [] # Stores metrics from steps if needed
        self.max_steps = max_steps
        self.current_step = 0 # Tracks total PSO steps within an episode
        self.last_gbest = float('inf') # Stores gbest from the previous step/turn
        self.adaptive_nt = adaptive_nt
        self.nt_range = nt_range
        self.steps_since_last_action = 0 # Counter for steps within an agent turn

        # Store convergence parameters to pass to PSO
        self.convergence_patience = convergence_patience
        self.convergence_threshold_gbest = convergence_threshold_gbest
        self.convergence_threshold_pbest_std = convergence_threshold_pbest_std

        # Initialize _current_nt
        self._current_nt = agent_step_size if not self.adaptive_nt else self.nt_range[0]

        # --- PSO Setup ---
        self.obj_fn = obj_func
        self.strategy = LocalBestStrategy(neighborhood_size=2) # Example strategy
        # Create PSO instance, passing convergence parameters
        self.pso = PSO(
            self.obj_fn,
            self.strategy,
            use_velocity_clamping=True, v_clamp_ratio=0.2, # Example PSO params
            convergence_patience=self.convergence_patience,
            convergence_threshold_gbest=self.convergence_threshold_gbest,
            convergence_threshold_pbest_std=self.convergence_threshold_pbest_std
        )
        # Provide swarm reference to strategy if it needs it
        if hasattr(self.pso.kb_sharing_strat, 'swarm'):
             self.pso.kb_sharing_strat.swarm = self.pso

        # --- Metrics Calculation ---
        self.metrics_calculator = SwarmMetrics()

        # === Observation Space ===
        # [squashed_norm_avg_vel, feasible_ratio, stable_ratio, percent_completion]
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            shape=(4,),
            dtype=np.float32
        )

        # === Action Space ===
        # omega ∈ [0.3, 1.0], c1, c2 ∈ [0.0, 3.0]
        self.action_bounds_low = np.array([0.3, 0.0, 0.0], dtype=np.float32)
        self.action_bounds_high = np.array([1.0, 3.0, 3.0], dtype=np.float32)
        self.action_dim = 3

        # Add nt to action space if adaptive
        if self.adaptive_nt:
            self.action_bounds_low = np.append(self.action_bounds_low, -1.0).astype(np.float32)
            self.action_bounds_high = np.append(self.action_bounds_high, 1.0).astype(np.float32)
            self.action_dim = 4

        # Agent's action space (outputs values in [-1, 1])
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

    def _rescale_action(self, action: np.ndarray) -> tuple:
        """
        Rescales the agent's action (output in [-1, 1]) to the actual
        parameter bounds (omega, c1, c2, and optionally nt).
        """
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
        """
        Resets the environment for a new episode.
        """
        super().reset(seed=seed)
        self.history = []
        self.current_step = 0
        self.steps_since_last_action = 0
        self._current_nt = self.nt_range[0] if self.adaptive_nt else self._current_nt

        # Re-initialize PSO state and convergence tracking
        self.pso = PSO(
            self.obj_fn,
            self.strategy,
            use_velocity_clamping=True, v_clamp_ratio=0.2,
            convergence_patience=self.convergence_patience,
            convergence_threshold_gbest=self.convergence_threshold_gbest,
            convergence_threshold_pbest_std=self.convergence_threshold_pbest_std
        )
        if hasattr(self.pso.kb_sharing_strat, 'swarm'):
             self.pso.kb_sharing_strat.swarm = self.pso

        # Evaluate initial positions and set initial gbest
        initial_gbest = float('inf')
        initial_metrics = {}
        if self.pso.particles:
            for p in self.pso.particles:
                fitness = self.obj_fn.evaluate(p.position)
                if fitness < p.pbest_value:
                    p.pbest_value = fitness
                    p.pbest_position = p.position.copy()
                if fitness < initial_gbest:
                    initial_gbest = fitness
                    self.pso.gbest_position = p.position.copy()

            self.pso.gbest_value = initial_gbest
            self.last_gbest = initial_gbest
            initial_metrics = self.metrics_calculator.compute(self.pso.particles, self.pso.bounds)
        else:
             self.last_gbest = float('inf')
             self.pso.gbest_value = float('inf')

        # Reset PSO's internal convergence tracking state
        self.pso.reset_convergence_tracking()

        observation = self._get_obs(initial_metrics)
        info = {}
        return observation, info

    def step(self, action: np.ndarray):
        """
        Executes one agent step (running `nt` internal PSO steps) and checks for termination.
        """
        # 1. Rescale action
        omega, c1, c2, next_nt = self._rescale_action(action)

        # 2. Update next nt if adaptive
        if self.adaptive_nt:
             self._current_nt = next_nt

        # 3. Initialize variables
        cumulative_reward = 0.0
        terminated = False # Standard Gym flag for episode end (goal reached, convergence, or failure)
        truncated = False # Standard Gym flag for episode end due to external limit (e.g., time)
        steps_taken_this_turn = 0
        step_metrics_list = []

        # Use the CPs determined at the start of this turn
        current_omega, current_c1, current_c2 = omega, c1, c2

        # 4. Run internal PSO loop for `self._current_nt` steps
        for _ in range(self._current_nt):
            # Check step limits *before* running the PSO step
            if self.current_step >= self.max_steps:
                truncated = True # Hit time limit
                print(f"Episode truncated at step {self.current_step} (max_steps reached).")
                break

            # --- Execute one PSO Step ---
            # Capture the 'converged' flag returned by the modified PSO class
            step_metrics, current_gbest, converged_this_step = self.pso.optimize_step(current_omega, current_c1, current_c2)
            self.current_step += 1
            steps_taken_this_turn += 1

            # --- Calculate Reward ---
            step_reward = self._calculate_relative_reward(current_gbest)
            self.last_gbest = current_gbest
            cumulative_reward += step_reward

            # --- Calculate and Store Detailed Metrics ---
            detailed_metrics = step_metrics.copy()
            try:
                swarm_state_metrics = compute_swarm_metrics(self.pso.particles)
                detailed_metrics['diversity'] = swarm_state_metrics.get('diversity', np.nan)
                if 'avg_velocity_magnitude' not in detailed_metrics and 'velocity' in swarm_state_metrics:
                     detailed_metrics['avg_velocity_magnitude'] = swarm_state_metrics['velocity']
                elif 'avg_velocity' in detailed_metrics and 'avg_velocity_magnitude' not in detailed_metrics:
                     detailed_metrics['avg_velocity_magnitude'] = detailed_metrics['avg_velocity']
            except Exception as e:
                print(f"Warning: Error calculating swarm metrics at step {self.current_step}: {e}")
                detailed_metrics['diversity'] = np.nan
                detailed_metrics['avg_velocity_magnitude'] = np.nan

            detailed_metrics['gbest_value'] = current_gbest
            detailed_metrics['omega'] = current_omega
            detailed_metrics['c1'] = current_c1
            detailed_metrics['c2'] = current_c2
            detailed_metrics['pso_step'] = self.current_step - 1
            step_metrics_list.append(detailed_metrics)

            # --- Check Termination Conditions ---
            # Check PSO convergence flag *first*
            if converged_this_step:
                terminated = True # Episode ends due to convergence
                print(f"PSO Swarm converged at step {self.current_step}.")
                break # Exit the inner loop for this agent turn

            # Check max steps limit again (in case the last step reached it)
            # Note: This check is slightly redundant due to the check at the loop start,
            # but ensures consistency if max_steps is exactly reached on the last iteration.
            if self.current_step >= self.max_steps:
                truncated = True # Hit time limit
                # Avoid double printing if already truncated at loop start
                if not any(m.get('pso_step', -1) == self.max_steps -1 for m in step_metrics_list):
                     print(f"Episode truncated at step {self.current_step} (max_steps reached).")
                break

        # 5. Prepare Observation
        final_metrics_for_obs = self.metrics_calculator.compute(self.pso.particles, self.pso.bounds)
        observation = self._get_obs(final_metrics_for_obs)

        # 6. Prepare Info Dictionary
        info = {
            'steps_taken': steps_taken_this_turn,
            'nt': self._current_nt,
            'step_metrics': step_metrics_list
        }

        # Ensure terminated and truncated are boolean
        terminated = bool(terminated)
        truncated = bool(truncated)

        # Return using standard Gym API v26+ style (terminated OR truncated signals episode end)
        return observation, cumulative_reward, terminated, truncated, info

    def _calculate_relative_reward(self, current_gbest: float) -> float:
        """
        Calculates the relative reward based on the change in global best solution.
        (Implementation based on Equation 25 from the paper)
        """
        y_old = self.last_gbest
        y_new = current_gbest

        if np.isclose(y_new, y_old) or not np.isfinite(y_new):
            return 0.0
        elif not np.isfinite(y_old) and np.isfinite(y_new): # Found finite solution
            return 1.0

        reward = 0.0
        if y_new > 0 and y_old > 0:
            beta = y_old + y_new
            shifted_old = y_old + beta
            shifted_new = y_new + beta
            if not np.isclose(shifted_old, 0):
                reward = 2.0 * (shifted_old - shifted_new) / shifted_old
        elif y_new < 0 and y_old < 0:
            beta = abs(y_old) + abs(y_new)
            shift = 2 * beta
            shifted_old = y_old + shift
            shifted_new = y_new + shift
            if not np.isclose(shifted_old, 0):
                reward = 2.0 * (shifted_old - shifted_new) / shifted_old
        elif y_new < 0 and y_old > 0:
            reward = 1.0

        return np.clip(reward, 0.0, 1.0)

    def _get_obs(self, metrics) -> np.ndarray:
         """
         Calculates the observation vector for the agent based on swarm metrics.
         """
         avg_vel = metrics.get('avg_velocity', 0.0)
         feasible_ratio = metrics.get('feasible_ratio', 0.0)
         stable_ratio = metrics.get('stable_ratio', 0.0)

         # Normalize average velocity using tanh squashing (Eq 22 approach)
         l, u = self.pso.bounds
         range_width = u - l
         if np.isclose(range_width, 0):
              normalized_vel_component = 0.0
         else:
             normalized_vel_component = math.tanh((avg_vel - 0) / (0.5 * range_width))

         squashed_norm_avg_vel = (normalized_vel_component + 1.0) / 2.0

         # Calculate percentage completion
         percent_completion = self.current_step / self.max_steps

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
        pass

    def close(self):
        pass
