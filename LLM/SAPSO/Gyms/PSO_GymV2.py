# --- Imports ---
import gym
import numpy as np
import math

from LLM.PSO.PSO import PSO
from LLM.PSO.ObjectiveFunctions.Training.Rastrgin import RastriginFunction # Example function
from LLM.PSO.Cognitive.LBest import LocalBestStrategy
# Assuming compute_swarm_metrics is available, e.g., from SwarmMetrics
from LLM.SwarmMetrics import SwarmMetrics, compute_swarm_metrics

class PSOEnv(gym.Env):
    """
    Gym environment for PSO adapted to align more closely with the paper.
    # --- ADDED ---
    - Calculates and returns detailed metrics for each internal PSO step.
    """
    # Define metadata for Gym compatibility (optional but good practice)
    metadata = {'render_modes': [], 'render_fps': 4}

    def __init__(self,
                 obj_func=RastriginFunction(dim=30, num_particles=300),
                 max_steps=5000,
                 agent_step_size=10, # Renamed from nt for clarity when fixed
                 adaptive_nt=False,  # Flag to toggle adaptive nt
                 nt_range=(1, 100)   # Range for adaptive nt (min_nt, max_nt)
                 ):
        """
        Initializes the PSO Environment.

        Args:
            obj_func: An instance of an objective function class (e.g., RastriginFunction).
            max_steps (int): Maximum number of PSO steps allowed per episode.
            agent_step_size (int): The fixed number of PSO steps to run between agent actions
                                   if adaptive_nt is False.
            adaptive_nt (bool): If True, the agent learns the number of steps (nt)
                                between actions. If False, uses agent_step_size.
            nt_range (tuple): The (min, max) allowed values for nt if adaptive_nt is True.
        """
        super().__init__()
        self.history = [] # Stores metrics from steps if needed
        self.max_steps = max_steps
        self.current_step = 0 # Tracks total PSO steps within an episode
        self.last_gbest = float('inf') # Stores gbest from the previous step/turn
        self.adaptive_nt = adaptive_nt
        self.nt_range = nt_range
        self.steps_since_last_action = 0 # Counter for steps within an agent turn

        # Initialize _current_nt: the number of steps for the *next* agent turn.
        # If fixed mode, it's set by agent_step_size.
        # If adaptive mode, it starts at the minimum of the range.
        self._current_nt = agent_step_size if not self.adaptive_nt else self.nt_range[0]

        # --- PSO Setup ---
        self.obj_fn = obj_func
        self.strategy = LocalBestStrategy(neighborhood_size=2) # Example strategy
        # Ensure PSO object is created correctly
        self.pso = PSO(self.obj_fn, self.strategy, use_velocity_clamping=True, v_clamp_ratio=0.2)
        # Provide swarm reference to strategy if it needs it (like GBest)
        if hasattr(self.pso.kb_sharing_strat, 'swarm'):
             self.pso.kb_sharing_strat.swarm = self.pso

        # --- Metrics Calculation ---
        # Use the SwarmMetrics class instance to calculate standard metrics
        self.metrics_calculator = SwarmMetrics()

        # === Observation Space ===
        # [squashed_norm_avg_vel, feasible_ratio, stable_ratio, percent_completion]
        # All values are intended to be in [0, 1]
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            shape=(4,),
            dtype=np.float32
        )

        # === Action Space ===
        # Define the bounds for the actual PSO parameters
        # omega ∈ [0.3, 1.0], c1, c2 ∈ [0.0, 3.0]
        self.action_bounds_low = np.array([0.3, 0.0, 0.0], dtype=np.float32)
        self.action_bounds_high = np.array([1.0, 3.0, 3.0], dtype=np.float32)
        self.action_dim = 3 # Base dimension for omega, c1, c2

        # If adaptive_nt is enabled, add nt to the action space
        if self.adaptive_nt:
            # The agent outputs nt action in [-1, 1], will be rescaled later
            self.action_bounds_low = np.append(self.action_bounds_low, -1.0).astype(np.float32)
            self.action_bounds_high = np.append(self.action_bounds_high, 1.0).astype(np.float32)
            self.action_dim = 4 # Dimension is now 4

        # Define the agent's action space (always outputs values in [-1, 1])
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

    def _rescale_action(self, action: np.ndarray) -> tuple:
        """
        Rescales the agent's action (output in [-1, 1]) to the actual
        parameter bounds (omega, c1, c2, and optionally nt).

        Args:
            action (np.ndarray): The action output by the SAC agent.

        Returns:
            tuple: A tuple containing (omega, c1, c2, nt).
                   'nt' is the rescaled value if adaptive_nt is True,
                   otherwise it's the current fixed value (_current_nt).
        """
        # Rescale omega, c1, c2 (first 3 components of action)
        low_cp = self.action_bounds_low[:3]
        high_cp = self.action_bounds_high[:3]
        # Formula: low + (0.5 * (action + 1.0) * (high - low))
        rescaled_cp = low_cp + (action[:3] + 1.0) * 0.5 * (high_cp - low_cp)
        # Clip to ensure bounds are strictly enforced
        rescaled_cp = np.clip(rescaled_cp, low_cp, high_cp)
        omega, c1, c2 = rescaled_cp

        # Determine nt value
        nt = self._current_nt # Default to current value (relevant for fixed mode)
        if self.adaptive_nt:
            # Rescale the 4th action component if nt is adaptive
            action_nt = action[3] # The nt action part is in [-1, 1]
            low_nt, high_nt = self.nt_range
            # Rescale action_nt from [-1, 1] to [low_nt, high_nt]
            rescaled_nt = low_nt + (action_nt + 1.0) * 0.5 * (high_nt - low_nt)
            # Clip and round to the nearest integer for the number of steps
            nt = int(np.round(np.clip(rescaled_nt, low_nt, high_nt)))

        return omega, c1, c2, nt

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state for a new episode.

        Args:
            seed (int, optional): Seed for the random number generator. Defaults to None.
            options (dict, optional): Additional options for resetting. Defaults to None.

        Returns:
            tuple: A tuple containing the initial observation and an info dictionary.
        """
        super().reset(seed=seed) # Handles seeding for reproducibility
        self.history = []
        self.current_step = 0
        self.steps_since_last_action = 0
        # Reset _current_nt correctly based on mode
        # If adaptive, reset to min range. If fixed, keep the value set in __init__.
        self._current_nt = self.nt_range[0] if self.adaptive_nt else self._current_nt

        # Re-initialize PSO state
        # Ensure obj_func is re-initialized if it has internal state (depends on function)
        # Example: self.obj_fn = RastriginFunction(...)
        self.pso = PSO(self.obj_fn, self.strategy, use_velocity_clamping=True, v_clamp_ratio=0.2)
        if hasattr(self.pso.kb_sharing_strat, 'swarm'):
             self.pso.kb_sharing_strat.swarm = self.pso

        # Evaluate initial positions and set initial gbest
        initial_gbest = float('inf')
        initial_metrics = {} # Dictionary to hold metrics for initial observation
        if self.pso.particles: # Ensure particles exist
            for p in self.pso.particles:
                # Calculate fitness only once per particle
                fitness = self.obj_fn.evaluate(p.position)
                # Update personal best (pbest)
                if fitness < p.pbest_value:
                    p.pbest_value = fitness
                    p.pbest_position = p.position.copy()
                # Update global best (gbest)
                if fitness < initial_gbest:
                    initial_gbest = fitness
                    # Store the position associated with the best fitness found so far
                    self.pso.gbest_position = p.position.copy()

            # Set the gbest value for the PSO instance
            self.pso.gbest_value = initial_gbest
            self.last_gbest = initial_gbest # Initialize last_gbest for reward calculation

            # Calculate initial metrics needed for the first observation
            initial_metrics = self.metrics_calculator.compute(self.pso.particles, self.pso.bounds)
        else:
             # Handle case with no particles (should not happen with standard PSO init)
             self.last_gbest = float('inf')
             self.pso.gbest_value = float('inf')


        observation = self._get_obs(initial_metrics) # Get observation based on initial state
        info = {} # Standard Gym API requires returning an info dict
        return observation, info

    def step(self, action: np.ndarray):
        """
        Executes one agent step (which involves running `nt` internal PSO steps).

        Args:
            action (np.ndarray): The action selected by the agent.

        Returns:
            tuple: A tuple containing:
                   - observation (np.ndarray): The observation after the steps.
                   - reward (float): The cumulative reward over the steps taken.
                   - terminated (bool): Whether the episode has ended (max_steps reached).
                   - truncated (bool): Whether the episode was cut short (e.g., time limit).
                   - info (dict): Additional information (steps_taken, nt, step_metrics).
        """
        # 1. Rescale agent action to get PSO parameters (omega, c1, c2) and the *next* nt
        omega, c1, c2, next_nt = self._rescale_action(action)

        # 2. Update the number of steps for the *next* turn if adaptive
        if self.adaptive_nt:
             self._current_nt = next_nt # Agent decides the *next* interval length

        # 3. Initialize variables for this agent turn
        cumulative_reward = 0.0
        terminated = False
        truncated = False # Gym standard: truncated usually means time limit, terminated means goal state
        steps_taken_this_turn = 0
        step_metrics_list = [] # Store detailed metrics for each internal PSO step

        # Use the CPs determined at the start of this turn for all internal steps
        current_omega, current_c1, current_c2 = omega, c1, c2

        # 4. Run internal PSO loop for `self._current_nt` steps
        for _ in range(self._current_nt):
            # Check if max steps reached before starting the PSO step
            if self.current_step >= self.max_steps:
                truncated = True # Set truncated flag as we hit the step limit
                break

            # --- Execute one PSO Step ---
            # optimize_step updates velocities, positions, pbest, and returns metrics & gbest
            step_metrics, current_gbest = self.pso.optimize_step(current_omega, current_c1, current_c2)
            self.current_step += 1 # Increment total PSO step counter *after* the step
            steps_taken_this_turn += 1

            # --- Calculate Reward for this PSO step ---
            step_reward = self._calculate_relative_reward(current_gbest)
            self.last_gbest = current_gbest # Update for the next step's reward calculation
            cumulative_reward += step_reward # Accumulate reward over the turn

            # --- Calculate and Store Detailed Metrics for Evaluation Plotting ---
            # Start with metrics returned by optimize_step (avg_vel, feasible, stable)
            detailed_metrics = step_metrics.copy() # Use copy to avoid modifying original

            # Calculate diversity using the external function
            # Ensure self.pso.particles reflects the state *after* optimize_step
            try:
                # compute_swarm_metrics might raise errors if particles are weird (e.g., NaN)
                swarm_state_metrics = compute_swarm_metrics(self.pso.particles)
                detailed_metrics['diversity'] = swarm_state_metrics.get('diversity', np.nan)
                 # Add avg velocity magnitude if not already present or named differently
                if 'avg_velocity_magnitude' not in detailed_metrics and 'velocity' in swarm_state_metrics:
                     detailed_metrics['avg_velocity_magnitude'] = swarm_state_metrics['velocity']
                elif 'avg_velocity' in detailed_metrics and 'avg_velocity_magnitude' not in detailed_metrics:
                     # Assuming avg_velocity from metrics_calculator is the magnitude
                     detailed_metrics['avg_velocity_magnitude'] = detailed_metrics['avg_velocity']

            except Exception as e:
                print(f"Warning: Error calculating swarm metrics at step {self.current_step}: {e}")
                detailed_metrics['diversity'] = np.nan
                detailed_metrics['avg_velocity_magnitude'] = np.nan


            # Add other relevant metrics
            detailed_metrics['gbest_value'] = current_gbest
            detailed_metrics['omega'] = current_omega # CP used for this step
            detailed_metrics['c1'] = current_c1
            detailed_metrics['c2'] = current_c2
            detailed_metrics['pso_step'] = self.current_step - 1 # Store 0-based step index

            step_metrics_list.append(detailed_metrics)

            # Check termination based on total steps AFTER incrementing
            if self.current_step >= self.max_steps:
                terminated = True # Set terminated flag (standard Gym)

        # 5. Prepare Observation for the Agent
        # The observation reflects the state *after* the last internal PSO step
        # Use the metrics calculator on the final particle state
        final_metrics_for_obs = self.metrics_calculator.compute(self.pso.particles, self.pso.bounds)
        observation = self._get_obs(final_metrics_for_obs)

        # 6. Prepare Info Dictionary
        info = {
            'steps_taken': steps_taken_this_turn, # Actual PSO steps executed this turn
            'nt': self._current_nt, # The number of steps *planned* for this turn (or next if adaptive)
            'step_metrics': step_metrics_list # Detailed metrics per PSO step
        }

        # Standard Gym requires terminated and truncated separated
        # terminated is True if max_steps is reached. truncated can be used for other limits.
        return observation, cumulative_reward, terminated, truncated, info

    def _calculate_relative_reward(self, current_gbest: float) -> float:
        """
        Calculates the relative reward based on the change in global best solution,
        aligned with Equation 25 from the paper.

        Args:
            current_gbest (float): The global best fitness value found in the current step.

        Returns:
            float: The calculated reward, clipped between 0.0 and 1.0.
        """
        y_old = self.last_gbest
        y_new = current_gbest

        # Handle edge case: no improvement
        if np.isclose(y_new, y_old):
            # If values are non-finite but equal (e.g., inf, inf), reward is 0
            if not np.isfinite(y_old) or not np.isfinite(y_new): return 0.0
            return 0.0 # No change, no reward

        # Handle edge case: one or both values are non-finite (e.g., inf)
        elif not np.isfinite(y_old) or not np.isfinite(y_new):
             # Improvement: Found a finite solution when the previous was infinite
             if np.isfinite(y_new) and not np.isfinite(y_old): return 1.0 # Max reward
             # No improvement or new value is also infinite
             else: return 0.0

        # Main calculation based on signs (Equation 25 logic)
        reward = 0.0 # Default reward
        if y_new > 0 and y_old > 0: # Both positive
            beta = y_old + y_new # Shift amount based on magnitudes
            # Shift values to be positive for relative calculation
            shifted_old = y_old + beta
            shifted_new = y_new + beta
            # Avoid division by zero if shifted_old becomes close to zero
            if np.isclose(shifted_old, 0): return 0.0
            # Calculate relative change of shifted values, scaled by 2
            reward = 2.0 * (shifted_old - shifted_new) / shifted_old
        elif y_new < 0 and y_old < 0: # Both negative
            beta = abs(y_old) + abs(y_new) # Shift amount based on magnitudes
            shift = 2 * beta # Ensure positivity after shift (as per paper figure caption)
            # Shift values to be positive
            shifted_old = y_old + shift
            shifted_new = y_new + shift
             # Avoid division by zero
            if np.isclose(shifted_old, 0): return 0.0
             # Calculate relative change, scaled by 2
            reward = 2.0 * (shifted_old - shifted_new) / shifted_old
        elif y_new < 0 and y_old > 0: # Improvement: Crossed zero from positive to negative
            reward = 1.0 # Fixed max reward as per paper discussion
        # else: (y_new > 0 and y_old < 0) or one is zero.
        # This implies worsening or no change across zero, reward remains 0.

        # Clip reward to [0, 1] as per paper's implied range
        return np.clip(reward, 0.0, 1.0)


    def _get_obs(self, metrics) -> np.ndarray:
         """
         Calculates the observation vector for the agent based on swarm metrics.
         Uses tanh squashing for average velocity (Eq 22) and percentage completion.

         Args:
             metrics (dict): Dictionary containing swarm metrics (e.g., from SwarmMetrics.compute).

         Returns:
             np.ndarray: The observation vector (normalized).
         """
         # Safely get metrics from the dictionary
         avg_vel = metrics.get('avg_velocity', 0.0)
         feasible_ratio = metrics.get('feasible_ratio', 0.0)
         stable_ratio = metrics.get('stable_ratio', 0.0)

         # Normalize average velocity using Eq 22 approach (tanh squashing)
         l, u = self.pso.bounds # Get search space bounds
         range_width = u - l
         if np.isclose(range_width, 0):
              normalized_vel_component = 0.0
         else:
             # Normalize velocity relative to half the range width
             normalized_vel_component = (avg_vel - 0) / (0.5 * range_width)
             # Apply tanh squashing -> maps to [-1, 1]
             normalized_vel_component = math.tanh(normalized_vel_component)

         # Shift tanh output to [0, 1] for the observation space
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
             # Replace non-finite values (e.g., with 0 or boundary values)
             obs_vec = np.nan_to_num(obs_vec, nan=0.0, posinf=1.0, neginf=0.0)
             # Ensure values are still clipped after potential replacement
             obs_vec = np.clip(obs_vec, 0.0, 1.0)

         return obs_vec

    # --- Standard Gym methods (optional) ---
    def render(self, mode='human'):
        """(Optional) Renders the environment."""
        pass # Implement visualization if needed

    def close(self):
        """(Optional) Cleans up environment resources."""
        pass # Add cleanup if resources are allocated (e.g., external processes)

