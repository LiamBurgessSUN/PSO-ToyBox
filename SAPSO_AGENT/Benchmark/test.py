import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import collections
import os
import traceback  # For logging exceptions
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

from SAPSO_AGENT.SAPSO.RL.ActorCritic.Agent import SACAgent
from SAPSO_AGENT.SAPSO.Environment.Environment import Environment
from SAPSO_AGENT.SAPSO.Graphics.graphing import (
    plot_evaluation_parameters,
    plot_stable_particles,
    plot_infeasible_particles,
    plot_average_velocity,
    plot_swarm_diversity,
    plot_gbest_convergence,
    plot_final_gbest_per_function,
    plot_gbest_convergence_per_function
)

from SAPSO_AGENT.Logs.logger import *
from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.Testing.Loader import test_objective_function_classes
from SAPSO_AGENT.CONFIG import *

# Import the new SAPSO plotting functionality
from SAPSO_AGENT.SAPSO.Graphics.sapso_plotting import SAPSOPlotter
from SAPSO_AGENT.CONFIG import PLOT_ONLY_AVERAGES


def generate_timestamped_filename(base_name: str, extension: str = "png") -> str:
    """
    Generate a filename with timestamp and base name.
    
    Args:
        base_name: The base name for the file
        extension: File extension (default: "png")
    
    Returns:
        str: Timestamped filename in format "YYYYMMDD_HHMMSS_base_name.extension"
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{base_name}.{extension}"

# --- Main Testing Function (Accepts Arguments) ---
def test_agent(
        env_dim=ENV_DIM,
        env_particles=ENV_PARTICLES,
        env_max_steps=ENV_MAX_STEPS,
        agent_step_size=AGENT_STEP_SIZE,
        adaptive_nt_mode=ADAPTIVE_NT_MODE,
        nt_range=NT_RANGE,
        num_eval_runs=NUM_EVAL_RUNS,
        checkpoint_base_dir=CHECKPOINT_BASE_DIR,
        hidden_dim=256,
        gamma=1.0,
        tau=0.005,
        alpha=0.2,
        actor_lr=3e-4,
        critic_lr=3e-4,
        v_clamp_ratio=0.2,
        use_velocity_clamping=USE_VELOCITY_CLAMPING,
        convergence_patience=50,
        convergence_threshold_gbest=1e-8,
        convergence_threshold_pbest_std=1e-6,
        # stability_threshold removed as it's not used by PSOEnvVectorized init
):
    """Main function to load and evaluate the trained SAC agent using PSOEnvVectorized."""
    module_name = Path(__file__).stem

    if not test_objective_function_classes:
        log_error("The test_objective_function_classes list is empty. Cannot test.", module_name)
        return

    # --- Model Configuration ---
    log_header("--- Model Configuration ---", module_name)
    log_info(f"Load RL Model: {LOAD_RL_MODEL}", module_name)
    log_info(f"Use New Model: {USE_NEW_MODEL}", module_name)
    log_info(f"Model Name Prefix: {MODEL_NAME_PREFIX}", module_name)
    log_info(f"Model Version Suffix: {MODEL_VERSION_SUFFIX}", module_name)

    # --- Determine Model File Path ---
    if checkpoint_base_dir is None:
        script_dir = Path(__file__).parent
        project_root_fallback = script_dir.parents[1]
        checkpoint_base_dir = project_root_fallback / "SAPSO" / "checkpoints"
        log_warning(f"checkpoint_base_dir not provided, using default: {checkpoint_base_dir}", module_name)

    # Add 'test' subdirectory for all test outputs
    test_output_dir = Path(checkpoint_base_dir) / "test"
    os.makedirs(test_output_dir, exist_ok=True)

    mode_suffix = "adaptive_nt" if adaptive_nt_mode else f"fixed_nt{agent_step_size}"
    checkpoint_dir = test_output_dir / f"checkpoints_sapso_vectorized_{mode_suffix}"
    
    # Generate model filename based on CONFIG settings
    version_suffix = MODEL_VERSION_SUFFIX if MODEL_VERSION_SUFFIX else ""
    model_filename = f"{MODEL_NAME_PREFIX}_{mode_suffix}{version_suffix}"
    MODEL_TO_LOAD = checkpoint_dir / f"{model_filename}_final.pth"

    log_info(f"Attempting to load model: {MODEL_TO_LOAD}", module_name)
    
    # Check if model loading is required
    if LOAD_RL_MODEL:
        if not MODEL_TO_LOAD.exists():
            log_error(f"Model file not found at {MODEL_TO_LOAD}", module_name)
            log_error("Ensure training ran with the same config and checkpoint base dir is correct.", module_name)
            if not USE_NEW_MODEL:
                log_error("Cannot continue without loading model. Exiting.", module_name)
                return
            else:
                log_warning("USE_NEW_MODEL is True. Continuing with new model.", module_name)
    elif not USE_NEW_MODEL:
        log_warning("USE_NEW_MODEL is False but LOAD_RL_MODEL is False. Using new model.", module_name)

    # --- Initialize Environment (Placeholder) ---
    log_info("Creating temporary vectorized environment to get dimensions...", module_name)
    try:
        temp_obj_func = test_objective_function_classes[0](dim=env_dim)
        temp_env = Environment(
            obj_func=temp_obj_func,
            num_particles=env_particles,
            max_steps=env_max_steps,
            agent_step_size=agent_step_size,
            adaptive_nt=adaptive_nt_mode,
            nt_range=nt_range,
            v_clamp_ratio=v_clamp_ratio,
            use_velocity_clamping=use_velocity_clamping,
            convergence_patience=convergence_patience,
            convergence_threshold_gbest=convergence_threshold_gbest,
            convergence_threshold_pbest_std=convergence_threshold_pbest_std,
        )
        state_dim = temp_env.observation_space.shape[0] if temp_env.observation_space.shape else 0
        action_dim = temp_env.action_space.shape[0] if temp_env.action_space.shape else 0
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
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        gamma=gamma,
        tau=tau,
        alpha=alpha,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        device=device,
        adaptive_nt=adaptive_nt_mode
    )

    # --- Load the Trained Agent Model ---
    if LOAD_RL_MODEL and MODEL_TO_LOAD.exists():
        try:
            agent.load(str(MODEL_TO_LOAD))
            log_success(f"Successfully loaded trained agent from: {MODEL_TO_LOAD}", module_name)
        except Exception as e:
            log_error(f"Error loading agent model: {e}", module_name)
            log_error(traceback.format_exc(), module_name)
            if not USE_NEW_MODEL:
                log_error("Cannot continue without loading model. Exiting.", module_name)
                return
            else:
                log_warning("USE_NEW_MODEL is True. Continuing with new model.", module_name)
    elif LOAD_RL_MODEL and not MODEL_TO_LOAD.exists():
        log_warning(f"Model file not found: {MODEL_TO_LOAD}", module_name)
        if not USE_NEW_MODEL:
            log_error("Cannot continue without loading model. Exiting.", module_name)
            return
        else:
            log_warning("USE_NEW_MODEL is True. Continuing with new model.", module_name)
    else:
        log_info("Using new untrained model for testing.", module_name)

    log_header(f"Starting {num_eval_runs} deterministic evaluation runs per test function...", module_name)

    # --- Data Aggregation ---
    evaluation_data = collections.defaultdict(list)
    all_eval_rewards = []
    final_gbests_all_runs = []
    eval_start_time = time.time()
    current_run_id = 0  # Track run ID for metrics
    
    # --- Metrics Collection for Plotting ---
    metrics_collector = None  # Will be initialized with the first environment's metrics calculator
    accumulated_metrics = {}  # Store metrics data across all evaluation runs

    # --- Evaluation Loop ---
    for func_index, func_class in enumerate(test_objective_function_classes):
        func_name = func_class.__name__
        log_info(f"--- Evaluating Function {func_index + 1}/{len(test_objective_function_classes)}: {func_name} ---", module_name)
        func_rewards = []
        last_gbest_this_func = []

        for run in range(num_eval_runs):
            current_run_id += 1  # Increment run ID for each evaluation run
            log_debug(f"  Starting Evaluation Run {run + 1}/{num_eval_runs} for {func_name} (Run ID: {current_run_id})...", module_name)
            run_best_gbest = float('inf')

            try:
                current_dim = env_dim
                obj_func_instance = func_class(dim=current_dim)
                eval_env = Environment(
                    obj_func=obj_func_instance,
                    num_particles=env_particles,
                    max_steps=env_max_steps,
                    agent_step_size=agent_step_size,
                    adaptive_nt=adaptive_nt_mode,
                    nt_range=nt_range,
                    v_clamp_ratio=v_clamp_ratio,
                    use_velocity_clamping=use_velocity_clamping,
                    convergence_patience=convergence_patience,
                    convergence_threshold_gbest=convergence_threshold_gbest,
                    convergence_threshold_pbest_std=convergence_threshold_pbest_std,
                    run_id=current_run_id
                )
                
                # Initialize metrics collector with the first environment's metrics calculator
                if metrics_collector is None and hasattr(eval_env, 'pso') and hasattr(eval_env.pso, 'metrics_calculator'):
                    metrics_collector = eval_env.pso.metrics_calculator
                    log_info("Initialized metrics collector for plotting", module_name)
                
                # TODO eval seed effects
                # state, _ = eval_env.reset(seed=1000 + func_index * num_eval_runs + run)
                state, _ = eval_env.reset()
                run_best_gbest = float(eval_env.pso.gbest_value) if eval_env and hasattr(eval_env, 'pso') else float('inf')
            except Exception as e:
                log_error(f"  Error creating environment for run {run + 1} on {func_name}: {e}", module_name)
                log_error(traceback.format_exc(), module_name)
                continue

            run_reward = 0.0
            term, trunc = False, False

            while not term and not trunc:
                if isinstance(state, torch.Tensor):
                    state_np = state.cpu().numpy()
                else:
                    state_np = np.array(state, dtype=np.float32)

                action = agent.select_action(state_np, deterministic=True)
                eval_omega, eval_c1, eval_c2, _ = eval_env._rescale_action(action)
                params_for_turn = [float(eval_omega), float(eval_c1), float(eval_c2)]

                try:
                    # The Environment now automatically passes the function name and run ID to the PSO
                    # The metrics tracking is handled internally by the Environment and PSOSwarm
                    next_state, reward, term, trunc, info = eval_env.step(action)
                    turn_final_gbest = info.get('final_gbest', np.inf)
                    if np.isfinite(turn_final_gbest) and isinstance(turn_final_gbest, (int, float)):
                        run_best_gbest = min(run_best_gbest, float(turn_final_gbest))
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
                        stable_ratio = step_metric_dict.get('stability_ratio', np.nan)
                        infeasible_ratio = step_metric_dict.get('infeasible_ratio', np.nan)
                        avg_vel_mag = step_metric_dict.get('avg_current_velocity_magnitude', np.nan)
                        diversity = step_metric_dict.get('swarm_diversity', np.nan)
                        gbest_val = step_metric_dict.get('gbest_value', np.nan)

                        # --- ADD CHECK for stability_ratio ---
                        if not isinstance(stable_ratio, (int, float)) or not np.isfinite(stable_ratio):
                            log_warning(f"  Invalid stability_ratio at step {pso_step_index}: {stable_ratio} (type: {type(stable_ratio)}). Storing NaN.", module_name)
                            stable_ratio = np.nan # Ensure NaN is stored if invalid
                        # --- END CHECK ---

                        # --- ENHANCED DEBUG LOG ---
                        is_diversity_finite = np.isfinite(diversity) if diversity is not None else False
                        if pso_step_index % 500 == 0: # Log occasionally
                            log_debug(f"  Extracted Diversity at step {pso_step_index}: {diversity} (type: {type(diversity)}, finite: {is_diversity_finite})", module_name)
                            # Also log stability ratio for comparison
                            log_debug(f"  Extracted Stability Ratio at step {pso_step_index}: {stable_ratio} (type: {type(stable_ratio)})", module_name)
                        # --- END DEBUG LOG ---

                        data_point = params_for_turn + [
                            stable_ratio, # Use potentially corrected value
                            infeasible_ratio,
                            avg_vel_mag,
                            diversity,
                            gbest_val
                        ]
                        evaluation_data[pso_step_index].append(data_point)


                state = next_state
                run_reward += reward

            # End of evaluation run
            func_rewards.append(run_reward)
            final_gbest_for_log = run_best_gbest
            if not np.isfinite(final_gbest_for_log) and eval_env and hasattr(eval_env, 'pso'):
                final_gbest_for_log = eval_env.pso.gbest_value

            if np.isfinite(final_gbest_for_log):
                last_gbest_this_func.append(final_gbest_for_log)

            log_debug(f"  Finished Run {run + 1}/{num_eval_runs}. Reward: {run_reward:.4f}, Final GBest: {final_gbest_for_log:.6e}", module_name)
            
            # Store metrics data from this evaluation run for plotting
            if eval_env and hasattr(eval_env, 'pso') and hasattr(eval_env.pso, 'metrics_calculator'):
                episode_metrics = eval_env.pso.metrics_calculator
                if episode_metrics is not None and hasattr(episode_metrics, 'metric_tracking') and episode_metrics.metric_tracking:
                    # Accumulate metrics data
                    for func_name, data in episode_metrics.metric_tracking.items():
                        if func_name not in accumulated_metrics:
                            accumulated_metrics[func_name] = data
                        else:
                            # Merge data from multiple episodes
                            for metric_name, values in data.items():
                                if metric_name in accumulated_metrics[func_name]:
                                    existing = accumulated_metrics[func_name][metric_name]
                                    if isinstance(existing, list) and isinstance(values, list):
                                        existing.extend(values)
                                    elif isinstance(existing, dict) and isinstance(values, dict):
                                        for k, v in values.items():
                                            if k in existing and isinstance(existing[k], list) and isinstance(v, list):
                                                existing[k].extend(v)
                                            elif k in existing and isinstance(existing[k], dict) and isinstance(v, dict):
                                                for k2, v2 in v.items():
                                                    if k2 in existing[k] and isinstance(existing[k][k2], list) and isinstance(v2, list):
                                                        existing[k][k2].extend(v2)
                                                    else:
                                                        existing[k][k2] = v2
                                                else:
                                                    existing[k] = v
                                        else:
                                            accumulated_metrics[func_name][metric_name] = values
                                    else:
                                        accumulated_metrics[func_name][metric_name] = values.copy() if isinstance(values, list) else values

                    # Also accumulate parameter data
                    if hasattr(episode_metrics, 'parameter_tracking') and episode_metrics.parameter_tracking:
                        for func_name, data in episode_metrics.parameter_tracking.items():
                            if func_name not in accumulated_metrics:
                                accumulated_metrics[func_name] = data
                            else:
                                # Merge parameter data
                                for param_name, values in data.items():
                                    if param_name in accumulated_metrics[func_name]:
                                        existing = accumulated_metrics[func_name][param_name]
                                        if isinstance(existing, list) and isinstance(values, list):
                                            existing.extend(values)
                                        elif isinstance(existing, dict) and isinstance(values, dict):
                                            for k, v in values.items():
                                                if k in existing and isinstance(existing[k], list) and isinstance(v, list):
                                                    existing[k].extend(v)
                                                elif k in existing and isinstance(existing[k], dict) and isinstance(v, dict):
                                                    for k2, v2 in v.items():
                                                        if k2 in existing[k] and isinstance(existing[k][k2], list) and isinstance(v2, list):
                                                            existing[k][k2].extend(v2)
                                                        else:
                                                            existing[k][k2] = v2
                                                else:
                                                    existing[k] = v
                                        else:
                                            accumulated_metrics[func_name][param_name] = values
                                    else:
                                        accumulated_metrics[func_name][param_name] = values.copy() if isinstance(values, list) else values
            
            try:
                eval_env.close()
            except Exception as e:
                log_warning(f"  Error closing environment for run {run + 1}: {e}", module_name)

        # Aggregate results for this function
        if func_rewards:
            finite_rewards = [r for r in func_rewards if np.isfinite(r)]
            if finite_rewards:
                all_eval_rewards.extend(finite_rewards)
                log_info(f"--- Function {func_name} Avg Reward ({len(finite_rewards)} valid runs): {np.mean(finite_rewards):.4f} +/- {np.std(finite_rewards):.4f} ---", module_name)
            else:
                log_warning(f"--- Function {func_name}: No valid rewards recorded ---", module_name)
        if last_gbest_this_func:
            final_gbests_all_runs.extend(last_gbest_this_func)
            log_info(f"--- Function {func_name} Avg Final GBest ({len(last_gbest_this_func)} valid runs): {np.mean(last_gbest_this_func):.6e} +/- {np.std(last_gbest_this_func):.6e} ---", module_name)
        else:
            log_warning(f"--- Function {func_name}: No valid final GBest values recorded ---", module_name)

    # --- End of all evaluation runs ---
    eval_end_time = time.time()
    log_header(f"Finished {num_eval_runs * len(test_objective_function_classes)} total evaluation runs.", module_name)
    log_info(f"Total evaluation time: {eval_end_time - eval_start_time:.2f} seconds.", module_name)

    # --- Calculate and Print Overall Final Statistics ---
    log_header("--- Overall Evaluation Results ---", module_name)
    if all_eval_rewards:
        mean_reward = np.mean(all_eval_rewards)
        std_reward = np.std(all_eval_rewards)
        log_info(f"Overall Mean Reward ({len(all_eval_rewards)} valid Runs): {mean_reward:.4f} (μ) +/- {std_reward:.4f} (σ)", module_name)
    else:
        log_warning("No valid evaluation reward data collected.", module_name)

    # --- Calculate NORMALIZED GBest Statistics ---
    if final_gbests_all_runs:
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

            log_info(f"Overall Mean NORMALIZED Final GBest ({len(final_gbests_all_runs)} valid Runs): {mean_normalized_gbest:.6f} (μ) +/- {std_normalized_gbest:.6f} (σ)", module_name)
            log_info(f"Overall NORMALIZED Final GBest Variance: {var_normalized_gbest:.6f}", module_name)
            mean_raw_gbest = np.mean(gbest_array)
            std_raw_gbest = np.std(gbest_array)
            log_info(f"(Raw Mean Final GBest: {mean_raw_gbest:.6e} +/- {std_raw_gbest:.6e})", module_name)
        else:
            log_warning(f"All valid final GBest values are nearly identical ({min_gbest:.6e}). Normalization skipped.", module_name)
            log_info(f"Raw Mean Final GBest: {min_gbest:.6e} +/- 0.0", module_name)
    else:
        log_warning("No final global best data collected for normalization.", module_name)
    log_info("---------------------------------", module_name)

    # --- Generate ALL Evaluation Plots ---
    if evaluation_data:
        log_info("Generating evaluation plots for test functions...", module_name)
        plot_output_dir = test_output_dir / "test_plots_vectorized"
        os.makedirs(plot_output_dir, exist_ok=True)
        plot_prefix = f"{model_filename}_TESTING_VECTORIZED"

        try:
            plot_evaluation_parameters(evaluation_data, env_max_steps, str(plot_output_dir), plot_prefix)
            plot_stable_particles(evaluation_data, env_max_steps, str(plot_output_dir), plot_prefix) # Index 3
            plot_infeasible_particles(evaluation_data, env_max_steps, str(plot_output_dir), plot_prefix)
            plot_average_velocity(evaluation_data, env_max_steps, str(plot_output_dir), plot_prefix)
            plot_swarm_diversity(evaluation_data, env_max_steps, str(plot_output_dir), plot_prefix)
            plot_gbest_convergence(evaluation_data, env_max_steps, str(plot_output_dir), plot_prefix)
            plot_final_gbest_per_function(evaluation_data, env_max_steps, str(plot_output_dir), plot_prefix)
            
            # Create function names list for plotting
            function_names = [func_class.__name__ for func_class in test_objective_function_classes]
            plot_gbest_convergence_per_function(evaluation_data, env_max_steps, str(plot_output_dir), plot_prefix, function_names)
            
            log_success(f"Evaluation plots saved in: {plot_output_dir}", module_name)
        except Exception as e:
            log_error(f"Error generating plots: {e}", module_name)
            log_error(traceback.format_exc(), module_name)
            log_warning("Plot generation failed. Ensure graphing.py is compatible.", module_name)
    else:
        log_warning("No evaluation data was collected, skipping plot generation.", module_name)

    # --- Generate SAPSO Metric Plots ---
    log_header("Generating SAPSO metric plots for evaluation", module_name)
    
    if accumulated_metrics:
        try:
            # Create a mock metrics calculator with accumulated data
            class MockMetricsCalculator:
                def __init__(self, metric_tracking, parameter_tracking):
                    self.metric_tracking = metric_tracking
                    self.parameter_tracking = parameter_tracking
            
            # Separate metric and parameter tracking data
            metric_tracking = {}
            parameter_tracking = {}
            
            for func_name, data in accumulated_metrics.items():
                # Check if this is metric data or parameter data
                if 'avg_step_size' in data or 'swarm_diversity' in data:
                    metric_tracking[func_name] = data
                if 'omega' in data or 'c1' in data:
                    parameter_tracking[func_name] = data
            
            if metric_tracking or parameter_tracking:
                mock_metrics = MockMetricsCalculator(metric_tracking, parameter_tracking)
                plotter = SAPSOPlotter(str(test_output_dir), plot_only_averages=PLOT_ONLY_AVERAGES)
                plotter.plot_all_metrics(mock_metrics, save_plots=True, show_plots=False)
                log_success("SAPSO metric plots generated successfully for evaluation", module_name)
            else:
                log_warning("No valid metrics or parameter data found in accumulated data", module_name)
        except Exception as e:
            log_error(f"Error generating SAPSO metric plots: {e}", module_name)
            log_error(traceback.format_exc(), module_name)
    else:
        log_warning("No metrics data available for plotting. Metrics collection may not be properly configured.", module_name)
        log_info("To enable metrics plotting, ensure the PSO environment is properly configured with metrics tracking.", module_name)

    # --- Plot Evaluation Reward Curve (Average across functions per 'epoch') ---
    avg_rewards_per_func = {}
    for func_index, func_class in enumerate(test_objective_function_classes):
        func_name = func_class.__name__
        # Calculate average only from finite rewards for that function
        # For evaluation, we can use the rewards collected during evaluation
        func_rewards = [r for r in all_eval_rewards if np.isfinite(r)]
        if func_rewards:
            avg_rewards_per_func[func_name] = np.mean(func_rewards)
        # else: # Optionally handle functions with no valid rewards
        #     avg_rewards_per_func[func_name] = np.nan

    if avg_rewards_per_func:
        plt.figure(figsize=(12, 6))
        # Filter out potential NaN values if some functions had no valid rewards
        valid_func_names = [name for name, avg in avg_rewards_per_func.items() if np.isfinite(avg)]
        valid_avg_rewards = [avg for avg in avg_rewards_per_func.values() if np.isfinite(avg)]

        if valid_func_names:
            plt.bar(valid_func_names, valid_avg_rewards)
            plt.title(f"SAC Avg Reward per Function (Evaluation) ({num_eval_runs} runs each) {mode_suffix}")
            plt.xlabel("Objective Function")
            plt.ylabel("Average Evaluation Reward")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save the plot in the test directory
            timestamped_filename = generate_timestamped_filename(f"{model_filename}_eval_rewards_per_func_static")
            plot_path = test_output_dir / timestamped_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            log_success(f"Evaluation reward plot saved to: {plot_path}", module_name)
        else:
            log_warning("No valid function names for plotting.", module_name)
    else:
        log_warning("No average rewards calculated for plotting.", module_name)

    return evaluation_data, all_eval_rewards, final_gbests_all_runs

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SAPSO agent with enhanced metrics tracking and plotting")
    
    # Environment parameters
    parser.add_argument("--env-dim", type=int, default=ENV_DIM, help="Environment dimension")
    parser.add_argument("--env-particles", type=int, default=ENV_PARTICLES, help="Number of particles")
    parser.add_argument("--env-max-steps", type=int, default=ENV_MAX_STEPS, help="Maximum steps per episode")
    parser.add_argument("--agent-step-size", type=int, default=AGENT_STEP_SIZE, help="Agent step size")
    parser.add_argument("--adaptive-nt-mode", type=str, default=str(ADAPTIVE_NT_MODE), help="Adaptive NT mode (true/false)")
    parser.add_argument("--nt-range", type=str, default=str(NT_RANGE), help="NT range as string '[min, max]'")
    parser.add_argument("--num-eval-runs", type=int, default=NUM_EVAL_RUNS, help="Number of evaluation runs")
    parser.add_argument("--checkpoint-base-dir", type=str, default=CHECKPOINT_BASE_DIR, help="Checkpoint base directory")
    
    # Agent hyperparameters
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma")
    parser.add_argument("--tau", type=float, default=0.005, help="Tau")
    parser.add_argument("--alpha", type=float, default=0.2, help="Alpha")
    parser.add_argument("--actor-lr", type=float, default=3e-4, help="Actor learning rate")
    parser.add_argument("--critic-lr", type=float, default=3e-4, help="Critic learning rate")
    
    # PSO parameters
    parser.add_argument("--v-clamp-ratio", type=float, default=0.2, help="Velocity clamp ratio")
    parser.add_argument("--use-velocity-clamping", type=str, default=str(USE_VELOCITY_CLAMPING), help="Use velocity clamping (true/false)")
    parser.add_argument("--convergence-patience", type=int, default=50, help="Convergence patience")
    parser.add_argument("--convergence-threshold-gbest", type=float, default=1e-8, help="Convergence threshold GBest")
    parser.add_argument("--convergence-threshold-pbest-std", type=float, default=1e-6, help="Convergence threshold PBest std")
    
    args = parser.parse_args()
    
    # Parse string arguments
    adaptive_nt_mode = args.adaptive_nt_mode.lower() == "true"
    use_velocity_clamping = args.use_velocity_clamping.lower() == "true"
    
    # Parse NT range
    try:
        nt_range_str = args.nt_range.strip("[]").split(",")
        nt_range = (int(nt_range_str[0]), int(nt_range_str[1]))
    except:
        nt_range = NT_RANGE
    
    # Run testing
    test_agent(
        env_dim=args.env_dim,
        env_particles=args.env_particles,
        env_max_steps=args.env_max_steps,
        agent_step_size=args.agent_step_size,
        adaptive_nt_mode=adaptive_nt_mode,
        nt_range=nt_range,
        num_eval_runs=args.num_eval_runs,
        checkpoint_base_dir=args.checkpoint_base_dir,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        v_clamp_ratio=args.v_clamp_ratio,
        use_velocity_clamping=use_velocity_clamping,
        convergence_patience=args.convergence_patience,
        convergence_threshold_gbest=args.convergence_threshold_gbest,
        convergence_threshold_pbest_std=args.convergence_threshold_pbest_std
    )
