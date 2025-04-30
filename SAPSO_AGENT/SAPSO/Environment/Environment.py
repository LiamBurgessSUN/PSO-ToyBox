# File: PSO-ToyBox/SAPSO_AGENT/SAPSO/Environment/Environment.py
# Refactored to align with updated PsoVectorized and SwarmMetricsVectorized,
# ensuring metrics used for observation match paper definitions where possible.

# --- Imports ---
import gymnasium as gym
import numpy as np
import math
import traceback
from pathlib import Path

from SAPSO_AGENT.SAPSO.PSO.Cognitive.GBest import GlobalBestStrategy

# --- Import Logger ---
from SAPSO_AGENT.Logs.logger import log_info, log_error, log_warning, log_success, log_header, log_debug

# --- Project Imports ---
# Use the updated PsoVectorized class ID if it changed
from SAPSO_AGENT.SAPSO.PSO.PsoVectorized import PSOVectorized
from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.Training.Functions.Rastrgin import RastriginFunction

# Metrics class is instantiated within PsoVectorized now, no direct import needed here usually
# but keep for clarity if needed elsewhere.


# --- Module Name for Logging ---
module_name = Path(__file__).stem  # Gets 'PSO_Gym_Vectorized'


class Environment(gym.Env):
    """
    Environment for PSO using the vectorized PSO implementation (PSOVectorized).
    Uses updated metrics aligned with the research paper.
    """

    def __init__(self,
                 obj_func=RastriginFunction(dim=30),
                 num_particles=30,
                 max_steps=5000,
                 agent_step_size=10,
                 adaptive_nt=False,
                 nt_range=(1, 100),
                 v_clamp_ratio=0.2,
                 use_velocity_clamping=True,
                 convergence_patience=50,
                 convergence_threshold_gbest=1e-8,
                 convergence_threshold_pbest_std=1e-6,
                 # stability_threshold removed as Poli's condition is used inside metrics
                 ):
        """
        Initializes the Vectorized PSO Environment.
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
        # self.stability_threshold = stability_threshold # Removed

        # --- Initialize PSOVectorized ---
        try:
            self.strategy = GlobalBestStrategy(None)
            self.pso = PSOVectorized(
                objective_function=self.obj_fn,
                num_particles=self.num_particles,
                strategy=self.strategy,
                v_clamp_ratio=self.v_clamp_ratio,
                use_velocity_clamping=self.use_velocity_clamping,
                convergence_patience=self.convergence_patience,
                convergence_threshold_gbest=self.convergence_threshold_gbest,
                convergence_threshold_pbest_std=self.convergence_threshold_pbest_std,
                # stability_threshold is now handled within metrics if needed, or implicitly via Poli's
            )
            self.strategy.swarm = self.pso
            self.last_gbest = self.pso.gbest_value
            log_info(f"PSOVectorized instance created successfully. Initial gbest: {self.last_gbest:.4e}", module_name)
        except Exception as e:
            log_error(f"Failed to initialize PSOVectorized: {e}", module_name)
            log_error(traceback.format_exc(), module_name)
            raise

        # === Observation Space ===
        # [squashed_norm_avg_vel, feasible_ratio, stability_ratio (Poli), percent_completion]
        # Note: stability_ratio now refers to Poli's condition metric
        obs_low = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=obs_low, high=obs_high, shape=(4,), dtype=np.float32
        )
        log_debug(f"Observation space defined: Box(low={obs_low}, high={obs_high})", module_name)

        # === Action Space === (No change needed here)
        self.action_bounds_low = np.array([0.3, 0.0, 0.0], dtype=np.float32)
        self.action_bounds_high = np.array([1.0, 3.0, 3.0], dtype=np.float32)
        self.action_dim = 3
        if self.adaptive_nt:
            self.action_bounds_low = np.append(self.action_bounds_low, -1.0).astype(np.float32)
            self.action_bounds_high = np.append(self.action_bounds_high, 1.0).astype(np.float32)
            self.action_dim = 4
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        log_debug(f"Action space defined: Box(low=-1.0, high=1.0, shape=({self.action_dim},))", module_name)

    def _rescale_action(self, action: np.ndarray) -> tuple:
        """ Rescales agent action [-1, 1] to actual parameter bounds. (No change needed)"""
        low_cp = self.action_bounds_low[:3]
        high_cp = self.action_bounds_high[:3]
        clipped_action_cp = np.clip(action[:3], -1.0, 1.0)
        rescaled_cp = low_cp + (clipped_action_cp + 1.0) * 0.5 * (high_cp - low_cp)
        rescaled_cp = np.clip(rescaled_cp, low_cp, high_cp)
        omega, c1, c2 = rescaled_cp

        nt = self._current_nt
        if self.adaptive_nt:
            action_nt = np.clip(action[3], -1.0, 1.0)
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
        self._current_nt = self.nt_range[0] if self.adaptive_nt else self._current_nt

        # Re-initialize PSOVectorized state
        try:
            self.pso = PSOVectorized(  # Re-initialize PSO
                objective_function=self.obj_fn,
                num_particles=self.num_particles,
                strategy=self.strategy,
                v_clamp_ratio=self.v_clamp_ratio,
                use_velocity_clamping=self.use_velocity_clamping,
                convergence_patience=self.convergence_patience,
                convergence_threshold_gbest=self.convergence_threshold_gbest,
                convergence_threshold_pbest_std=self.convergence_threshold_pbest_std,
            )
            self.last_gbest = self.pso.gbest_value
            log_info(f"Environment reset complete. Initial gbest: {self.last_gbest:.4e}", module_name)
        except Exception as e:
            log_error(f"Failed to re-initialize PSOVectorized during reset: {e}", module_name)
            log_error(traceback.format_exc(), module_name)
            default_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return default_obs, {'error': 'PSO reset failed'}

        # Get initial metrics for observation
        # Cannot calculate step size or stability ratio at step 0
        initial_metrics = {}
        if self.pso.metrics_calculator and self.pso.num_particles > 0:
            try:
                # Calculate metrics that don't need previous step or CPs
                vel_mag = np.mean(np.linalg.norm(self.pso.velocities, axis=1))
                l, u = self.pso.bounds
                is_out = np.any((self.pso.positions < l) | (self.pso.positions > u), axis=1)
                infeasible = np.sum(is_out) / self.pso.num_particles

                initial_metrics = {
                    'avg_current_velocity_magnitude': vel_mag,
                    'infeasible_ratio': infeasible,
                    'stability_ratio': 0.0,  # Cannot determine stability at step 0
                    # Add others if needed by _get_obs and calculable
                }
            except Exception as e:
                log_warning(f"Failed to compute subset of initial metrics during reset: {e}", module_name)
                initial_metrics = {}  # Ensure it's a dict

        observation = self._get_obs(initial_metrics)
        info = {'initial_gbest': self.pso.gbest_value}
        return observation, info

    def step(self, action: np.ndarray):
        """ Executes one agent step (running `nt` internal PSO steps). """
        log_debug(f"Step {self.current_step}: Received action: {action}", module_name)
        omega, c1, c2, next_nt = self._rescale_action(action)

        if self.adaptive_nt:
            self._current_nt = next_nt
            log_debug(f"Adaptive nt: Set next nt to {self._current_nt}", module_name)

        cumulative_reward = 0.0
        terminated = False
        truncated = False
        steps_taken_this_turn = 0
        step_metrics_list = []

        current_omega, current_c1, current_c2 = omega, c1, c2  # Use fixed CPs for this agent turn
        log_debug(f"Running {self._current_nt} internal PSO steps with w={omega:.3f}, c1={c1:.3f}, c2={c2:.3f}",
                  module_name)

        final_metrics_for_obs = {}  # Store metrics from the *last* internal step for observation

        for i in range(self._current_nt):
            if self.current_step >= self.max_steps:
                truncated = True
                log_info(f"Episode truncated at step {self.current_step} (max_steps reached).", module_name)
                break

            try:
                # pso.optimize_step now returns metrics aligned with paper definitions
                step_metrics, current_gbest, converged_this_step = self.pso.optimize_step(
                    current_omega, current_c1, current_c2
                )
                self.current_step += 1
                steps_taken_this_turn += 1
                final_metrics_for_obs = step_metrics  # Update metrics for final observation
            except Exception as e:
                log_error(f"Error during pso.optimize_step at step {self.current_step}: {e}", module_name)
                log_error(traceback.format_exc(), module_name)
                terminated = True
                step_metrics = {}
                current_gbest = self.last_gbest
                converged_this_step = False
                final_metrics_for_obs = {}  # Reset metrics on error
                log_warning("Terminating episode due to error in optimize_step.", module_name)
                break

            step_reward = self._calculate_relative_reward(current_gbest)
            self.last_gbest = current_gbest
            cumulative_reward += step_reward

            # Add CPs used to the metrics dict for logging/analysis if needed
            log_metrics = step_metrics.copy()
            log_metrics['omega'] = current_omega
            log_metrics['c1'] = current_c1
            log_metrics['c2'] = current_c2
            log_metrics['pso_step'] = self.current_step - 1
            step_metrics_list.append(log_metrics)
            log_debug(
                f"  PSO Step {self.current_step}: gbest={current_gbest:.4e}, reward={step_reward:.4f}, converged={converged_this_step}, stability={step_metrics.get('stability_ratio', 'N/A')}",
                module_name)

            if converged_this_step:
                terminated = True
                log_info(f"PSO Swarm converged at step {self.current_step}.", module_name)
                break

            if self.current_step >= self.max_steps:
                truncated = True
                if not terminated:
                    log_info(f"Episode truncated at step {self.current_step} (max_steps reached).", module_name)
                break

        # 5. Prepare Observation using metrics from the *last successful* internal step
        observation = self._get_obs(final_metrics_for_obs)

        # 6. Prepare Info Dictionary
        info = {
            'steps_taken': steps_taken_this_turn,
            'nt': self._current_nt,
            'step_metrics': step_metrics_list,
            'final_gbest': self.pso.gbest_value
        }

        terminated = bool(terminated)
        truncated = bool(truncated)
        log_debug(
            f"Agent turn end. Steps taken: {steps_taken_this_turn}. Terminated: {terminated}. Truncated: {truncated}. Reward: {cumulative_reward:.4f}",
            module_name)

        return observation, cumulative_reward, terminated, truncated, info

    def _calculate_relative_reward(self, current_gbest: float) -> float:
        """ Calculates reward based on relative improvement in gbest. (No change needed)"""
        y_old = self.last_gbest
        y_new = current_gbest

        if not np.isfinite(y_new) or np.isclose(y_new, y_old):
            log_debug(f"Reward calc: No change or non-finite new gbest ({y_new:.4e}). Reward=0.", module_name)
            return 0.0
        if not np.isfinite(y_old) and np.isfinite(y_new):
            log_debug(f"Reward calc: Found first finite gbest ({y_new:.4e}). Reward=1.0.", module_name)
            return 1.0

        reward = 0.0
        beta = np.abs(y_old) + np.abs(y_new) + 1e-9  # Add epsilon to avoid beta=0 if both y_old,y_new=0

        if y_new >= 0 and y_old >= 0:
            shifted_old = y_old + beta
            shifted_new = y_new + beta
            if not np.isclose(shifted_old, 0):
                reward = 2.0 * (shifted_old - shifted_new) / shifted_old
            log_debug(f"Reward calc (Case +/+): y_old={y_old:.4e}, y_new={y_new:.4e}, reward={reward:.4f}", module_name)
        elif y_new < 0 and y_old < 0:
            shift = 2 * beta
            shifted_old = y_old + shift
            shifted_new = y_new + shift
            if not np.isclose(shifted_old, 0):
                reward = 2.0 * (shifted_old - shifted_new) / shifted_old
            log_debug(f"Reward calc (Case -/-): y_old={y_old:.4e}, y_new={y_new:.4e}, reward={reward:.4f}", module_name)
        elif y_new < 0 and y_old >= 0:
            reward = 1.0
            log_debug(f"Reward calc (Case +/-): y_old={y_old:.4e}, y_new={y_new:.4e}, reward=1.0", module_name)
        else:
            log_debug(f"Reward calc (Case -/+): y_old={y_old:.4e}, y_new={y_new:.4e}, reward=0.0", module_name)
            reward = 0.0

        final_reward = np.clip(reward, 0.0, 1.0)
        if not np.isclose(final_reward, reward):
            log_warning(f"Reward calculation ({reward:.4f}) clipped to {final_reward:.4f}.", module_name)
        return final_reward

    def _get_obs(self, metrics: dict) -> np.ndarray:
        """
        Calculates the observation vector from swarm metrics, aligned with paper.
        Uses: [squashed_norm_avg_vel, feasible_ratio, stability_ratio (Poli), percent_completion]
        Velocity normalization calculation updated to conform closer to Eq. 22 structure.
        """
        # Use the average *current* velocity magnitude (scalar) for the velocity component
        avg_vel = metrics.get('avg_current_velocity_magnitude', 0.0)
        # Use infeasible ratio to get feasible ratio
        infeasible_ratio = metrics.get('infeasible_ratio', 1.0)  # Default to 1.0 (0% feasible) if missing
        feasible_ratio = 1.0 - infeasible_ratio
        # Use stability ratio based on Poli's condition
        stable_ratio = metrics.get('stability_ratio', 0.0)  # Default to 0.0 (unstable) if missing

        # --- Updated Velocity Normalization (closer to Eq. 22 structure, applied to avg_vel) ---
        l, u = self.pso.bounds
        range_width = u - l
        squashed_norm_avg_vel = 0.0 # Default value

        if np.isclose(range_width, 0):
            # Handle case where search space range is zero (e.g., 1D problem with bounds [5, 5])
            # A velocity of 0 should map to 0.5 in the [0, 1] range.
            # Any non-zero velocity in this case is technically infinite relative to the range.
            # We map non-zero velocity to 1.0 (max value in [0,1]).
            squashed_norm_avg_vel = 0.5 if np.isclose(avg_vel, 0) else 1.0
            log_debug(f"Zero range width detected. Avg Vel: {avg_vel:.2e}. Normalized Vel: {squashed_norm_avg_vel}", module_name)
        else:
            # Calculate center and scaling factor based on Eq. 22 structure
            center = (u + l) / 2.0
            scale_factor = 2.0 / range_width # Equivalent to 1 / (0.5 * range_width)

            # Apply centering and scaling to the average velocity magnitude
            centered_scaled_avg_vel = scale_factor * (avg_vel - center)

            # Apply tanh squashing
            normalized_vel_component = math.tanh(centered_scaled_avg_vel) # Result in [-1, 1]

            # Map the [-1, 1] result to [0, 1] for the observation space
            squashed_norm_avg_vel = (normalized_vel_component + 1.0) / 2.0
            log_debug(f"Vel Norm: avg_vel={avg_vel:.3e}, centered_scaled={centered_scaled_avg_vel:.3f}, tanh={normalized_vel_component:.3f}, final={squashed_norm_avg_vel:.3f}", module_name)
        # --- End Updated Velocity Normalization ---

        # Calculate percentage completion
        percent_completion = self.current_step / self.max_steps if self.max_steps > 0 else 1.0

        # Construct the observation vector
        obs_vec = np.array([
            np.clip(squashed_norm_avg_vel, 0.0, 1.0), # Use the newly calculated velocity component
            np.clip(feasible_ratio, 0.0, 1.0),
            np.clip(stable_ratio, 0.0, 1.0),  # Use Poli's stability ratio
            np.clip(percent_completion, 0.0, 1.0)
        ], dtype=np.float32)

        if not np.all(np.isfinite(obs_vec)):
            log_warning(f"Non-finite values in observation at step {self.current_step}. Original: {obs_vec}",
                        module_name)
            obs_vec = np.nan_to_num(obs_vec, nan=0.0, posinf=1.0, neginf=0.0)
            obs_vec = np.clip(obs_vec, 0.0, 1.0)
            log_warning(f"Corrected observation: {obs_vec}", module_name)

        log_debug(f"Observation at step {self.current_step}: {obs_vec}", module_name)
        return obs_vec

    # --- Standard Gym methods (optional) ---
    def render(self, mode='human'):
        log_warning("Render method not implemented for PSOEnvVectorized.", module_name)
        pass

    def close(self):
        log_info("Closing PSOEnvVectorized.", module_name)
        pass
