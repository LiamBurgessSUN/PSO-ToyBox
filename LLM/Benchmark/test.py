# test.py - Place in PSO-ToyBox/LLM/Benchmark/
# Uses static imports and accepts hyperparameters as arguments
# Calculates and prints NORMALIZED final gbest mean/std
# --- UPDATED TO USE PSOEnvVectorized ---
# --- UPDATED to move all imports to top ---
# --- UPDATED to normalize final gbest statistics ---
# --- UPDATED to use logger ---

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import random
import sys
import collections
import traceback  # For logging exceptions
from pathlib import Path

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
# Removed try-except block around these imports as requested
from LLM.RL.ActorCritic.Agent import SACAgent
from LLM.SAPSO.Gyms.PSO_Gym_Vectorized import PSOEnvVectorized  # Use Vectorized Env
from LLM.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction  # Base class needed
# Import plotting functions (assuming path is correct)
from LLM.SAPSO.graphics.graphing import (
    plot_evaluation_parameters,
    plot_stable_particles,
    plot_infeasible_particles,
    plot_average_velocity,
    plot_swarm_diversity,
    plot_gbest_convergence
)
# --- Static Imports for Testing Objective Functions ---
# (Imports remain the same)
from LLM.PSO.ObjectiveFunctions.Testing.CrossLegTable import CrossLegTableFunction
from LLM.PSO.ObjectiveFunctions.Testing.Lanczos import Lanczos3Function
from LLM.PSO.ObjectiveFunctions.Testing.Michalewicz import MichalewiczFunction
from LLM.PSO.ObjectiveFunctions.Testing.Schaffer import Schaffer4Function
from LLM.PSO.ObjectiveFunctions.Testing.SineEnvelope import SineEnvelopeFunction
from LLM.PSO.ObjectiveFunctions.Testing.StretchedVSineWave import StretchedVSineWaveFunction
from LLM.PSO.ObjectiveFunctions.Testing.Wavy import WavyFunction

# --- End Project Imports ---


# --- Manually Create List of Test Objective Function Classes ---
# (List remains the same)
test_objective_function_classes = [
    CrossLegTableFunction, Lanczos3Function, MichalewiczFunction, Schaffer4Function,
    SineEnvelopeFunction, StretchedVSineWaveFunction, WavyFunction
]


# --- End Manual List ---


# --- Main Testing Function (Accepts Arguments) ---
def test_agent(
        env_dim=30,
        env_particles=30,  # Keep this parameter
        env_max_steps=5000,
        agent_step_size=125,  # Must match trained model config
        adaptive_nt_mode=False,  # Must match trained model config
        nt_range=(1, 50),  # Must match trained model config
        num_eval_runs=30,
        checkpoint_base_dir=None,  # Expect this from benchmark.py
        # Agent HPs (only needed for init before load)
        hidden_dim=256,
        gamma=1.0,
        tau=0.005,
        alpha=0.2,
        actor_lr=3e-4,
        critic_lr=3e-4,
        # Add PSO/Env params if needed by PSOEnvVectorized constructor
        v_clamp_ratio=0.2,
        use_velocity_clamping=True,
        convergence_patience=50,  # These might not be used by test env, but good practice
        convergence_threshold_gbest=1e-8,
        convergence_threshold_pbest_std=1e-6,
        stability_threshold=1e-3
):
    """Main function to load and evaluate the trained SAC agent using PSOEnvVectorized."""
    # === Use Passed-in Hyperparameters ===
    module_name = Path(__file__).stem  # 'test'

    if not test_objective_function_classes:
        log_error("The test_objective_function_classes list is empty. Cannot test.", module_name)
        return

    # --- Determine Model File Path ---
    if checkpoint_base_dir is None:
        script_dir = Path(__file__).parent
        project_root_fallback = script_dir.parents[1]  # Adjust path if needed
        checkpoint_base_dir = project_root_fallback / "SAPSO" / "checkpoints"  # Example adjusted path
        log_warning(f"checkpoint_base_dir not provided, using default: {checkpoint_base_dir}", module_name)

    # Construct path based on training mode (must match train.py)
    mode_suffix = "adaptive_nt" if adaptive_nt_mode else f"fixed_nt{agent_step_size}"
    # Assuming model was saved in the vectorized checkpoints dir
    checkpoint_dir = Path(checkpoint_base_dir) / f"checkpoints_sapso_vectorized_{mode_suffix}"
    checkpoint_prefix = f"sac_psoenv_vectorized_{mode_suffix}"
    MODEL_TO_LOAD = checkpoint_dir / f"{checkpoint_prefix}_final.pth"  # Load the final model

    log_info(f"Attempting to load model: {MODEL_TO_LOAD}", module_name)
    if not MODEL_TO_LOAD.exists():
        log_error(f"Model file not found at {MODEL_TO_LOAD}", module_name)
        log_error(
            "Ensure that training was run with the same configuration (adaptive_nt/fixed_nt) and that the checkpoint base directory is correct.",
            module_name)
        return

    # --- Initialize Environment (Placeholder using PSOEnvVectorized) ---
    log_info("Creating temporary vectorized environment to get dimensions...", module_name)
    try:
        temp_obj_func = test_objective_function_classes[0](dim=env_dim)
        # --- Use PSOEnvVectorized ---
        temp_env = PSOEnvVectorized(
            obj_func=temp_obj_func,
            num_particles=env_particles,  # Pass num_particles
            max_steps=env_max_steps,
            agent_step_size=agent_step_size,
            adaptive_nt=adaptive_nt_mode,
            nt_range=nt_range,
            # Pass other relevant params
            v_clamp_ratio=v_clamp_ratio,
            use_velocity_clamping=use_velocity_clamping,
            convergence_patience=convergence_patience,
            convergence_threshold_gbest=convergence_threshold_gbest,
            convergence_threshold_pbest_std=convergence_threshold_pbest_std,
            stability_threshold=stability_threshold
        )
        state_dim = temp_env.observation_space.shape[0]
        action_dim = temp_env.action_space.shape[0]
        temp_env.close()
        log_info("Temporary environment closed.", module_name)
    except Exception as e:
        log_error(f"Error creating temporary environment: {e}", module_name)
        log_error(traceback.format_exc(), module_name)
        return

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_header(f"--- Testing Configuration (Vectorized Env) ---", module_name)
    log_info(f"Using device: {device}", module_name)
    log_info(f"Loading Model Trained With:", module_name)
    log_info(f"  Adaptive Nt Mode: {adaptive_nt_mode}", module_name)
    if not adaptive_nt_mode:
        log_info(f"  Fixed Agent Step Size (Nt): {agent_step_size}", module_name)
    else:
        log_info(f"  Adaptive Nt Range: {nt_range}", module_name)
    log_info(f"Test Functions: {len(test_objective_function_classes)}", module_name)
    log_info(f"Evaluation Runs per Function: {num_eval_runs}", module_name)
    log_info(f"-------------------------------------------", module_name)

    # --- Initialize Agent ---
    agent = SACAgent(
        state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim,
        gamma=gamma, tau=tau, alpha=alpha, actor_lr=actor_lr, critic_lr=critic_lr,
        device=device, adaptive_nt=adaptive_nt_mode  # Use argument
    )

    # --- Load the Trained Agent Model ---
    try:
        agent.load(str(MODEL_TO_LOAD))
        log_success(f"Successfully loaded trained agent from: {MODEL_TO_LOAD}", module_name)
    except Exception as e:
        log_error(f"Error loading agent model: {e}", module_name)
        log_error(traceback.format_exc(), module_name)
        return

    # === Use passed-in Evaluation Parameters ===
    log_header(f"Starting {num_eval_runs} deterministic evaluation runs per test function...", module_name)

    # --- Data Aggregation ---
    # evaluation_data format: {pso_step: [data_point_run1, data_point_run2, ...], ...}
    # data_point = [w, c1, c2, stable%, infeasible%, avg_vel_mag, diversity, gbest]
    evaluation_data = collections.defaultdict(list)
    all_eval_rewards = []
    final_gbests_all_runs = []  # Store final gbest from each run
    eval_start_time = time.time()

    # --- Evaluation Loop ---
    for func_index, func_class in enumerate(test_objective_function_classes):
        func_name = func_class.__name__
        log_info(f"--- Evaluating Function {func_index + 1}/{len(test_objective_function_classes)}: {func_name} ---",
                 module_name)
        func_rewards = []
        last_gbest_this_func = []  # Store final gbest for runs on this function

        for run in range(num_eval_runs):
            log_debug(f"  Starting Evaluation Run {run + 1}/{num_eval_runs} for {func_name}...", module_name)
            # Track the best gbest found *during* this specific run
            run_best_gbest = np.inf

            try:
                current_dim = env_dim
                # Add dimension checks if needed (e.g., for Giunta)
                # if func_name == "GiuntaFunction": current_dim = 2

                obj_func_instance = func_class(dim=current_dim)
                # --- Use PSOEnvVectorized ---
                eval_env = PSOEnvVectorized(
                    obj_func=obj_func_instance,
                    num_particles=env_particles,  # Pass num_particles
                    max_steps=env_max_steps,
                    agent_step_size=agent_step_size,
                    adaptive_nt=adaptive_nt_mode,
                    nt_range=nt_range,
                    # Pass other relevant params
                    v_clamp_ratio=v_clamp_ratio,
                    use_velocity_clamping=use_velocity_clamping,
                    convergence_patience=convergence_patience,
                    convergence_threshold_gbest=convergence_threshold_gbest,
                    convergence_threshold_pbest_std=convergence_threshold_pbest_std,
                    stability_threshold=stability_threshold
                )
                # Use different seeds for evaluation runs
                state, _ = eval_env.reset(seed=1000 + func_index * num_eval_runs + run)
                run_best_gbest = eval_env.pso.gbest_value  # Initialize with starting gbest

            except Exception as e:
                log_error(f"  Error creating environment for run {run + 1} on {func_name}: {e}", module_name)
                log_error(traceback.format_exc(), module_name)
                continue  # Skip this run

            run_reward = 0.0
            term, trunc = False, False

            while not term and not trunc:
                # Ensure state is numpy array
                if isinstance(state, torch.Tensor):
                    state_np = state.cpu().numpy()
                else:
                    state_np = np.array(state, dtype=np.float32)

                # Select action deterministically
                action = agent.select_action(state_np, deterministic=True)
                eval_omega, eval_c1, eval_c2, eval_nt = eval_env._rescale_action(action)
                params_for_turn = [float(eval_omega), float(eval_c1), float(eval_c2)]

                try:
                    next_state, reward, term, trunc, info = eval_env.step(action)
                    # Update run's best gbest using the final gbest from the agent turn
                    turn_final_gbest = info.get('final_gbest', np.inf)
                    if np.isfinite(turn_final_gbest):
                        run_best_gbest = min(run_best_gbest, turn_final_gbest)

                except Exception as e:
                    log_error(f"  Error during env.step() in run {run + 1} on {func_name}: {e}", module_name)
                    log_error(traceback.format_exc(), module_name)
                    log_warning("  Terminating run early.", module_name)
                    trunc = True
                    next_state = state
                    reward = 0
                    info = {}

                # --- Collect detailed metrics per PSO step for plotting ---
                step_metrics_list = info.get('step_metrics', [])

                for step_metric_dict in step_metrics_list:
                    pso_step_index = step_metric_dict.get('pso_step', -1)
                    if pso_step_index >= 0 and pso_step_index < eval_env.max_steps:
                        # Extract metrics using keys from SwarmMetricsVectorized/PSOEnvVectorized
                        stable_ratio = step_metric_dict.get('stability_ratio', np.nan)
                        feasible_ratio = step_metric_dict.get('feasible_ratio', np.nan)
                        infeasible_ratio = 1.0 - feasible_ratio if not np.isnan(feasible_ratio) else np.nan
                        avg_vel_mag = step_metric_dict.get('avg_velocity_magnitude', np.nan)
                        diversity = step_metric_dict.get('diversity', np.nan)
                        gbest_val = step_metric_dict.get('gbest_value', np.nan)

                        # Create data point list in the order expected by graphing.py
                        data_point = params_for_turn + [
                            stable_ratio, infeasible_ratio, avg_vel_mag, diversity, gbest_val
                        ]
                        evaluation_data[pso_step_index].append(data_point)

                state = next_state
                run_reward += reward

            # End of evaluation run
            func_rewards.append(run_reward)
            # Use the best gbest found during the run
            final_gbest_for_log = run_best_gbest
            # Fallback if tracking failed
            if not np.isfinite(final_gbest_for_log) and eval_env and hasattr(eval_env, 'pso'):
                final_gbest_for_log = eval_env.pso.gbest_value

            if np.isfinite(final_gbest_for_log):  # Only store valid final gbests
                last_gbest_this_func.append(final_gbest_for_log)

            log_debug(
                f"  Finished Run {run + 1}/{num_eval_runs}. Reward: {run_reward:.4f}, Final GBest: {final_gbest_for_log:.6e}",
                module_name)
            try:
                eval_env.close()
            except Exception as e:
                log_warning(f"  Error closing environment for run {run + 1}: {e}", module_name)

        # Aggregate rewards and final gbests for this function
        if func_rewards:
            # Filter out potential non-finite rewards
            finite_rewards = [r for r in func_rewards if np.isfinite(r)]
            if finite_rewards:
                all_eval_rewards.extend(finite_rewards)
                log_info(
                    f"--- Function {func_name} Avg Reward ({len(finite_rewards)} valid runs): {np.mean(finite_rewards):.4f} +/- {np.std(finite_rewards):.4f} ---",
                    module_name)
            else:
                log_warning(f"--- Function {func_name}: No valid rewards recorded ---", module_name)
        if last_gbest_this_func:  # Already filtered for finite values
            final_gbests_all_runs.extend(last_gbest_this_func)
            log_info(
                f"--- Function {func_name} Avg Final GBest ({len(last_gbest_this_func)} valid runs): {np.mean(last_gbest_this_func):.6e} +/- {np.std(last_gbest_this_func):.6e} ---",
                module_name)
        else:
            log_warning(f"--- Function {func_name}: No valid final GBest values recorded ---", module_name)

    # --- End of all evaluation runs ---
    eval_end_time = time.time()
    log_header(f"Finished {num_eval_runs * len(test_objective_function_classes)} total evaluation runs.", module_name)
    log_info(f"Total evaluation time: {eval_end_time - eval_start_time:.2f} seconds.", module_name)

    # --- Calculate and Print Overall Final Statistics ---
    log_header("--- Overall Evaluation Results ---", module_name)
    if all_eval_rewards:  # Already filtered for finite
        mean_reward = np.mean(all_eval_rewards)
        std_reward = np.std(all_eval_rewards)
        log_info(
            f"Overall Mean Reward ({len(all_eval_rewards)} valid Runs): {mean_reward:.4f} (μ) +/- {std_reward:.4f} (σ)",
            module_name)
    else:
        log_warning("No valid evaluation reward data collected.", module_name)

    # --- Calculate NORMALIZED GBest Statistics ---
    if final_gbests_all_runs:  # Already filtered for finite
        gbest_array = np.array(final_gbests_all_runs)
        min_gbest = np.min(gbest_array)
        max_gbest = np.max(gbest_array)
        gbest_range = max_gbest - min_gbest

        log_info(f"Observed Final GBest Range (Test Set): [{min_gbest:.6e}, {max_gbest:.6e}]", module_name)

        if gbest_range > 1e-12:
            normalized_gbests = (gbest_array - min_gbest) / gbest_range
            mean_normalized_gbest = np.mean(normalized_gbests)
            std_normalized_gbest = np.std(normalized_gbests)
            var_normalized_gbest = np.var(normalized_gbests)

            log_info(
                f"Overall Mean NORMALIZED Final GBest ({len(final_gbests_all_runs)} valid Runs): {mean_normalized_gbest:.6f} (μ) +/- {std_normalized_gbest:.6f} (σ)",
                module_name)
            log_info(f"Overall NORMALIZED Final GBest Variance: {var_normalized_gbest:.6f}", module_name)
            mean_raw_gbest = np.mean(gbest_array)
            std_raw_gbest = np.std(gbest_array)
            log_info(f"(Raw Mean Final GBest: {mean_raw_gbest:.6e} +/- {std_raw_gbest:.6e})", module_name)
        else:
            log_warning(f"All valid final GBest values are nearly identical ({min_gbest:.6e}). Normalization skipped.",
                        module_name)
            log_info(f"Raw Mean Final GBest: {min_gbest:.6e} +/- 0.0", module_name)
    else:
        log_warning("No final global best data collected for normalization.", module_name)
    log_info("---------------------------------", module_name)

    # --- Generate ALL Evaluation Plots ---
    if evaluation_data:
        log_info("Generating evaluation plots for test functions...", module_name)
        # Save plots relative to the loaded model's directory
        plot_output_dir = checkpoint_dir / "test_plots_vectorized"
        os.makedirs(plot_output_dir, exist_ok=True)
        # Use the same prefix derived from the loaded model config
        plot_prefix = f"{checkpoint_prefix}_TESTING_VECTORIZED"

        # Call plotting functions (ensure graphing.py expects the data format)
        # The data format [w, c1, c2, stable%, infeasible%, avg_vel, diversity, gbest]
        # stored in evaluation_data should match what graphing.py expects.
        try:
            plot_evaluation_parameters(evaluation_data, env_max_steps, str(plot_output_dir), plot_prefix)
            plot_stable_particles(evaluation_data, env_max_steps, str(plot_output_dir), plot_prefix)
            plot_infeasible_particles(evaluation_data, env_max_steps, str(plot_output_dir), plot_prefix)
            plot_average_velocity(evaluation_data, env_max_steps, str(plot_output_dir), plot_prefix)
            plot_swarm_diversity(evaluation_data, env_max_steps, str(plot_output_dir), plot_prefix)
            plot_gbest_convergence(evaluation_data, env_max_steps, str(plot_output_dir), plot_prefix)
            log_success(f"Evaluation plots saved in: {plot_output_dir}", module_name)
        except Exception as e:
            log_error(f"Error generating plots: {e}", module_name)
            log_error(traceback.format_exc(), module_name)
            log_warning("Plot generation failed. Ensure graphing.py is compatible with the data format.", module_name)
    else:
        log_warning("No evaluation data was collected, skipping plot generation.", module_name)

# --- Main execution block removed ---
# This script is intended to be called by benchmark.py
