# File: PSO-ToyBox/LLM/SAPSO/Gyms/PSO_Gym_Vectorized.py
# Refactored to use the logger module

# --- Imports ---
import gymnasium as gym  # Use gymnasium instead of gym
import numpy as np
import math
import collections
import traceback  # For logging exceptions
from pathlib import Path  # To get module name

# --- Import Logger ---
# Using the specified import path: from LLM.Logs import logger
# Assuming Logs directory is two levels up from Gyms directory
try:
    from ...Logs import logger  # Adjust relative path if needed
    from ...Logs.logger import log_info, log_error, log_warning, log_success, log_header, log_debug
except ImportError:
    # Fallback print if logger fails to import
    print("ERROR: Logger module not found at 'LLM.Logs.logger'. Please check path.")
    print("Falling back to standard print statements.")


    # Define dummy functions
    def log_info(msg, mod):
        print(f"INFO [{mod}]: {msg}")


    def log_error(msg, mod):
        print(f"ERROR [{mod}]: {msg}")


    def log_warning(msg, mod):
        print(f"WARNING [{mod}]: {msg}")


    def log_success(msg, mod):
        print(f"SUCCESS [{mod}]: {msg}")


    def log_header(msg, mod):
        print(f"HEADER [{mod}]: {msg}")


    def log_debug(msg, mod):
        print(f"DEBUG [{mod}]: {msg}")  # Optional debug

# --- Project Imports ---
# Do not change existing imports as requested
from LLM.PSO.PsoVectorized import PSOVectorized
from LLM.PSO.ObjectiveFunctions.Training.Rastrgin import RastriginFunction  # Example function
from LLM.PSO.Cognitive.LBest import LocalBestStrategy  # Example strategy

# --- Module Name for Logging ---
module_name = Path(__file__).stem  # Gets 'PSO_Gym_Vectorized'


class PSOEnvVectorized(gym.Env):
    """
    Gym environment for PSO using the vectorized PSO implementation (PSOVectorized).
    Adapts PSO control parameters based on RL agent actions.
    Includes logging.
    """
    metadata = {'render_modes': [], 'render_fps': 4}

    def __init__(self,
                 obj_func=RastriginFunction(dim=30),  # Removed num_particles here
                 num_particles=30,  # Added num_particles parameter
                 max_steps=5000,
                 agent_step_size=10,
                 adaptive_nt=False,
                 nt_range=(1, 100),
                 v_clamp_ratio=0.2,  # Added PSO param
                 use_velocity_clamping=True,  # Added PSO param
                 convergence_patience=50,
                 convergence_threshold_gbest=1e-8,
                 convergence_threshold_pbest_std=1e-6,
                 stability_threshold=1e-3  # Added metrics param
                 ):
        """
        Initializes the Vectorized PSO Environment.
        Logs initialization parameters.
        """
        super().__init__()
        log_info(f"Initializing PSOEnvVectorized: adaptive_nt={adaptive_nt}, max_steps={max_steps}", module_name)
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
        try:
            # Strategy object is passed but PSOVectorized currently assumes G-Best
            self.strategy = LocalBestStrategy(neighborhood_size=2)  # Example strategy instance
            self.pso = PSOVectorized(
                objective_function=self.obj_fn,
                num_particles=self.num_particles,
                strategy=self.strategy,  # Pass strategy object
                v_clamp_ratio=self.v_clamp_ratio,
                use_velocity_clamping=self.use_velocity_clamping,
                convergence_patience=self.convergence_patience,
                convergence_threshold_gbest=self.convergence_threshold_gbest,
                convergence_threshold_pbest_std=self.convergence_threshold_pbest_std,
                stability_threshold=self.stability_threshold
            )
            self.last_gbest = self.pso.gbest_value  # Initialize last_gbest after PSO init
            log_info(f"PSOVectorized instance created successfully. Initial gbest: {self.last_gbest:.4e}", module_name)
        except Exception as e:
            log_error(f"Failed to initialize PSOVectorized: {e}", module_name)
            log_error(traceback.format_exc(), module_name)
            raise  # Re-raise exception as environment cannot function

        # === Observation Space ===
        # [squashed_norm_avg_vel, feasible_ratio, stability_ratio, percent_completion]
        obs_low = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=obs_low, high=obs_high, shape=(4,), dtype=np.float32
        )
        log_debug(f"Observation space defined: Box(low={obs_low}, high={obs_high})", module_name)

        # === Action Space ===
        self.action_bounds_low = np.array([0.3, 0.0, 0.0], dtype=np.float32)  # omega, c1, c2
        self.action_bounds_high = np.array([1.0, 3.0, 3.0], dtype=np.float32)
        self.action_dim = 3

        if self.adaptive_nt:
            self.action_bounds_low = np.append(self.action_bounds_low, -1.0).astype(
                np.float32)  # nt action range [-1, 1]
            self.action_bounds_high = np.append(self.action_bounds_high, 1.0).astype(np.float32)
            self.action_dim = 4

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        log_debug(f"Action space defined: Box(low=-1.0, high=1.0, shape=({self.action_dim},))", module_name)

    def _rescale_action(self, action: np.ndarray) -> tuple:
        """ Rescales agent action [-1, 1] to actual parameter bounds. """
        # Rescale omega, c1, c2
        low_cp = self.action_bounds_low[:3]
        high_cp = self.action_bounds_high[:3]
        # Clip action first to ensure it's within [-1, 1] before rescaling
        clipped_action_cp = np.clip(action[:3], -1.0, 1.0)
        rescaled_cp = low_cp + (clipped_action_cp + 1.0) * 0.5 * (high_cp - low_cp)
        # Clip again to handle potential floating point inaccuracies
        rescaled_cp = np.clip(rescaled_cp, low_cp, high_cp)
        omega, c1, c2 = rescaled_cp

        # Determine nt value
        nt = self._current_nt  # Default for fixed mode
        if self.adaptive_nt:
            action_nt = np.clip(action[3], -1.0, 1.0)  # Clip nt action
            low_nt, high_nt = self.nt_range
            rescaled_nt = low_nt + (action_nt + 1.0) * 0.5 * (high_nt - low_nt)
            nt = int(np.round(np.clip(rescaled_nt, low_nt, high_nt)))

        log_debug(f"Rescaled action: w={omega:.3f}, c1={c1:.3f}, c2={c2:.3f}, nt={nt}", module_name)
        return omega, c1, c2, nt

    def reset(self, seed=None, options=None):
        """ Resets the environment for a new episode. """
        super().reset(seed=seed)
        log_info(f"Resetting environment (Seed: {seed}). Current step: {self.current_step}", module_name)
        self.current_step = 0
        # Reset nt to default only if adaptive
        self._current_nt = self.nt_range[0] if self.adaptive_nt else self._current_nt

        # Re-initialize PSOVectorized state
        try:
            self.pso = PSOVectorized(
                objective_function=self.obj_fn,
                num_particles=self.num_particles,
                strategy=self.strategy,  # Pass strategy object
                v_clamp_ratio=self.v_clamp_ratio,
                use_velocity_clamping=self.use_velocity_clamping,
                convergence_patience=self.convergence_patience,
                convergence_threshold_gbest=self.convergence_threshold_gbest,
                convergence_threshold_pbest_std=self.convergence_threshold_pbest_std,
                stability_threshold=self.stability_threshold
            )
            self.last_gbest = self.pso.gbest_value  # Reset last_gbest
            log_info(f"Environment reset complete. Initial gbest: {self.last_gbest:.4e}", module_name)
        except Exception as e:
            log_error(f"Failed to re-initialize PSOVectorized during reset: {e}", module_name)
            log_error(traceback.format_exc(), module_name)
            # Return a default observation and info, or raise error
            default_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return default_obs, {'error': 'PSO reset failed'}

        # Get initial metrics from the PSO instance's calculator
        initial_metrics = {}
        if self.pso.metrics_calculator:
            try:
                initial_metrics = self.pso.metrics_calculator.compute(
                    self.pso.positions, self.pso.velocities, self.pso.bounds
                )
            except Exception as e:
                log_warning(f"Failed to compute initial metrics during reset: {e}", module_name)

        observation = self._get_obs(initial_metrics)
        info = {'initial_gbest': self.pso.gbest_value}  # Add initial gbest to info
        return observation, info

    def step(self, action: np.ndarray):
        """ Executes one agent step (running `nt` internal PSO steps). """
        log_debug(f"Step {self.current_step}: Received action: {action}", module_name)
        # 1. Rescale action
        omega, c1, c2, next_nt = self._rescale_action(action)

        # 2. Update next nt if adaptive
        if self.adaptive_nt:
            self._current_nt = next_nt
            log_debug(f"Adaptive nt: Set next nt to {self._current_nt}", module_name)

        # 3. Initialize variables for this agent turn
        cumulative_reward = 0.0
        terminated = False
        truncated = False
        steps_taken_this_turn = 0
        step_metrics_list = []  # To store metrics from each internal PSO step

        current_omega, current_c1, current_c2 = omega, c1, c2
        log_debug(f"Running {self._current_nt} internal PSO steps with w={omega:.3f}, c1={c1:.3f}, c2={c2:.3f}",
                  module_name)

        # 4. Run internal PSO loop for `self._current_nt` steps
        for i in range(self._current_nt):
            if self.current_step >= self.max_steps:
                truncated = True
                log_info(f"Episode truncated at step {self.current_step} (max_steps reached).", module_name)
                break

            # --- Execute one PSO Step ---
            try:
                step_metrics, current_gbest, converged_this_step = self.pso.optimize_step(
                    current_omega, current_c1, current_c2
                )
                self.current_step += 1
                steps_taken_this_turn += 1
            except Exception as e:
                log_error(f"Error during pso.optimize_step at step {self.current_step}: {e}", module_name)
                log_error(traceback.format_exc(), module_name)
                # Decide how to handle: terminate, truncate, or try to continue?
                # Terminating seems safest if the core optimization fails.
                terminated = True  # Mark as terminated due to internal error
                step_metrics = {}  # No valid metrics
                current_gbest = self.last_gbest  # Use last known gbest
                converged_this_step = False
                log_warning("Terminating episode due to error in optimize_step.", module_name)
                break  # Exit inner loop

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
            detailed_metrics_for_info['pso_step'] = self.current_step - 1  # Step index (0-based)
            step_metrics_list.append(detailed_metrics_for_info)
            log_debug(
                f"  PSO Step {self.current_step}: gbest={current_gbest:.4e}, reward={step_reward:.4f}, converged={converged_this_step}",
                module_name)

            # --- Check Termination Conditions ---
            if converged_this_step:
                terminated = True
                log_info(f"PSO Swarm converged at step {self.current_step}.", module_name)
                break

            # Check max steps limit again (redundant check, but safe)
            if self.current_step >= self.max_steps:
                truncated = True
                if not terminated:  # Avoid double logging if already terminated
                    log_info(f"Episode truncated at step {self.current_step} (max_steps reached).", module_name)
                break

        # 5. Prepare Observation for the *next* agent step
        final_metrics_for_obs = {}
        if self.pso.metrics_calculator:
            try:
                final_metrics_for_obs = self.pso.metrics_calculator.compute(
                    self.pso.positions, self.pso.velocities, self.pso.bounds
                )
            except Exception as e:
                log_warning(f"Failed to compute final metrics for observation: {e}", module_name)
                # Observation will use defaults (zeros) in _get_obs

        observation = self._get_obs(final_metrics_for_obs)

        # 6. Prepare Info Dictionary
        info = {
            'steps_taken': steps_taken_this_turn,
            'nt': self._current_nt,
            'step_metrics': step_metrics_list,  # List of metrics from each internal step
            'final_gbest': self.pso.gbest_value  # Gbest at the end of the agent turn
        }

        terminated = bool(terminated)
        truncated = bool(truncated)
        log_debug(
            f"Agent turn end. Steps taken: {steps_taken_this_turn}. Terminated: {terminated}. Truncated: {truncated}. Reward: {cumulative_reward:.4f}",
            module_name)

        return observation, cumulative_reward, terminated, truncated, info

    def _calculate_relative_reward(self, current_gbest: float) -> float:
        """ Calculates reward based on relative improvement in gbest. """
        y_old = self.last_gbest
        y_new = current_gbest

        # Handle non-finite cases or no change
        if not np.isfinite(y_new) or np.isclose(y_new, y_old):
            log_debug(f"Reward calc: No change or non-finite new gbest ({y_new:.4e}). Reward=0.", module_name)
            return 0.0
        if not np.isfinite(y_old) and np.isfinite(y_new):  # Found first finite solution
            log_debug(f"Reward calc: Found first finite gbest ({y_new:.4e}). Reward=1.0.", module_name)
            return 1.0

        # Apply relative reward formula (Equation 25 from paper)
        reward = 0.0
        beta = np.abs(y_old) + np.abs(y_new)

        if y_new >= 0 and y_old >= 0:  # Both non-negative
            shifted_old = y_old + beta
            shifted_new = y_new + beta
            if not np.isclose(shifted_old, 0):
                reward = 2.0 * (shifted_old - shifted_new) / shifted_old
            log_debug(f"Reward calc (Case +/+): y_old={y_old:.4e}, y_new={y_new:.4e}, reward={reward:.4f}", module_name)
        elif y_new < 0 and y_old < 0:  # Both negative
            shift = 2 * beta
            shifted_old = y_old + shift
            shifted_new = y_new + shift
            if not np.isclose(shifted_old, 0):
                reward = 2.0 * (shifted_old - shifted_new) / shifted_old
            log_debug(f"Reward calc (Case -/-): y_old={y_old:.4e}, y_new={y_new:.4e}, reward={reward:.4f}", module_name)
        elif y_new < 0 and y_old >= 0:  # Improvement crossed zero
            reward = 1.0
            log_debug(f"Reward calc (Case +/-): y_old={y_old:.4e}, y_new={y_new:.4e}, reward=1.0", module_name)
        else:  # Case y_new >= 0 and y_old < 0 (Worsened across zero)
            log_debug(f"Reward calc (Case -/+): y_old={y_old:.4e}, y_new={y_new:.4e}, reward=0.0", module_name)
            reward = 0.0

        # Ensure reward is non-negative and capped
        final_reward = np.clip(reward, 0.0, 1.0)
        if not np.isclose(final_reward, reward):
            log_warning(
                f"Reward calculation resulted in value outside [0,1] ({reward:.4f}). Clipped to {final_reward:.4f}.",
                module_name)
        return final_reward

    def _get_obs(self, metrics: dict) -> np.ndarray:
        """ Calculates the observation vector from swarm metrics. """
        # Use keys returned by SwarmMetricsVectorized
        avg_vel = metrics.get('avg_velocity_magnitude', 0.0)
        feasible_ratio = metrics.get('feasible_ratio', 1.0)  # Default to 1.0 if missing
        # Use 'stability_ratio' key from SwarmMetricsVectorized
        stable_ratio = metrics.get('stability_ratio', 1.0)  # Default to 1.0 if missing

        # Normalize average velocity using tanh squashing (Eq 22 adaptation)
        l, u = self.pso.bounds
        range_width = u - l
        squashed_norm_avg_vel = 0.0  # Default
        if np.isclose(range_width, 0):
            # Avoid division by zero, velocity is irrelevant if range is zero
            squashed_norm_avg_vel = 0.5  # Represents zero velocity in [-1,1] -> [0,1] mapping
        else:
            # Normalize velocity magnitude relative to half the search space width
            scale = 0.5 * range_width
            # tanh argument: how many "half-ranges" the velocity covers per step
            tanh_arg = avg_vel / scale if scale > 0 else avg_vel
            normalized_vel_component = math.tanh(tanh_arg)
            # Squash tanh output [-1, 1] to observation range [0, 1]
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
            log_warning(
                f"Non-finite values detected in observation at step {self.current_step}. Replacing/Clipping. Original: {obs_vec}",
                module_name)
            obs_vec = np.nan_to_num(obs_vec, nan=0.0, posinf=1.0, neginf=0.0)
            obs_vec = np.clip(obs_vec, 0.0, 1.0)
            log_warning(f"Corrected observation: {obs_vec}", module_name)

        log_debug(f"Observation at step {self.current_step}: {obs_vec}", module_name)
        return obs_vec

    # --- Standard Gym methods (optional) ---
    def render(self, mode='human'):
        log_warning("Render method not implemented for PSOEnvVectorized.", module_name)
        pass  # Implement visualization if needed

    def close(self):
        log_info("Closing PSOEnvVectorized.", module_name)
        pass  # Clean up resources if needed
