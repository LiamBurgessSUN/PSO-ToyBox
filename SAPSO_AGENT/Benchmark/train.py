import random
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import traceback  # For logging exceptions
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

from SAPSO_AGENT.SAPSO.RL.ActorCritic.Agent import SACAgent
from SAPSO_AGENT.SAPSO.RL.Replay.ReplayBuffer import ReplayBuffer
from SAPSO_AGENT.SAPSO.Environment.Environment import Environment

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.Training.Loader import objective_function_classes
from SAPSO_AGENT.Logs.logger import *
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


# --- Main Training Function (Accepts Arguments) ---
def train_agent(
        env_dim=ENV_DIM,
        env_particles=ENV_PARTICLES,  # Keep this parameter
        env_max_steps=ENV_MAX_STEPS,
        agent_step_size=AGENT_STEP_SIZE,
        adaptive_nt_mode=ADAPTIVE_NT_MODE,
        nt_range=NT_RANGE,
        episodes_per_function=EPISODES_PER_FUNCTION,
        batch_size=BATCH_SIZE,
        start_steps=START_STEPS,
        updates_per_step=UPDATES_PER_STEP,
        save_freq_multiplier=SAVE_FREQ_MULTIPLIER,
        checkpoint_base_dir=CHECKPOINT_BASE_DIR,
        # Agent HPs
        hidden_dim=256,
        gamma=1.0,
        tau=0.005,
        alpha=0.2,
        actor_lr=3e-4,
        critic_lr=3e-4,
        # Add PSO/Env params if needed by PSOEnvVectorized constructor
        v_clamp_ratio=0.2,
        use_velocity_clamping=USE_VELOCITY_CLAMPING,
        convergence_patience=50,
        convergence_threshold_gbest=1e-8,
        convergence_threshold_pbest_std=1e-6,
        stability_threshold=1e-3
):
    """Main function to train the SAC agent using PSOEnvVectorized."""
    # === Use Passed-in Hyperparameters ===
    module_name = Path(__file__).stem  # 'train'

    if not objective_function_classes:
        log_error("The objective_function_classes list is empty. Cannot train.", module_name)
        return  # Or raise an error

    # --- Initialize Environment (using a placeholder initially with PSOEnvVectorized) ---
    log_info("Creating temporary vectorized environment to get dimensions...", module_name)
    try:
        # Ensure num_particles is passed to the objective function constructor if needed
        temp_obj_func = objective_function_classes[0](dim=env_dim)  # num_particles not needed by func directly
        # --- Use PSOEnvVectorized ---
        temp_env = Environment(
            obj_func=temp_obj_func,
            num_particles=env_particles,  # Pass num_particles here
            max_steps=env_max_steps,
            agent_step_size=agent_step_size,
            adaptive_nt=adaptive_nt_mode,
            nt_range=nt_range,
            # Pass other relevant params if needed, using defaults otherwise
            v_clamp_ratio=v_clamp_ratio,
            use_velocity_clamping=use_velocity_clamping,
            convergence_patience=convergence_patience,
            convergence_threshold_gbest=convergence_threshold_gbest,
            convergence_threshold_pbest_std=convergence_threshold_pbest_std,
            # stability_threshold=stability_threshold
        )
        state_dim = temp_env.observation_space.shape[0] if temp_env.observation_space.shape else 0
        action_dim = temp_env.action_space.shape[0] if temp_env.action_space.shape else 0
        temp_env.close()
        log_info("Temporary environment closed.", module_name)
    except Exception as e:
        log_error(f"Error creating temporary environment: {e}", module_name)
        log_error(traceback.format_exc(), module_name)
        return  # Returning None or similar might be better than sys.exit

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_header(f"--- Training Configuration (Vectorized Env) ---", module_name)
    log_info(f"Using device: {device}", module_name)
    log_info(f"Objective Functions: {len(objective_function_classes)}", module_name)
    log_info(f"Env Dim: {env_dim}, Particles: {env_particles}, Max Steps: {env_max_steps}", module_name)
    log_info(f"Adaptive Nt Mode: {adaptive_nt_mode}", module_name)
    if not adaptive_nt_mode:
        log_info(f"Fixed Agent Step Size (Nt): {agent_step_size}", module_name)
    else:
        log_info(f"Adaptive Nt Range: {nt_range}", module_name)
    log_info(f"Episodes per Function: {episodes_per_function}", module_name)
    log_info(f"Batch Size: {batch_size}, Start Steps: {start_steps}", module_name)
    log_info(f"Updates Per Step: {updates_per_step}", module_name)
    log_info(f"Save Freq Multiplier (Functions): {save_freq_multiplier}", module_name)
    log_info(f"---------------------------------------------", module_name)

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

    # --- Model Load/Save Configuration ---
    log_header("--- Model Configuration ---", module_name)
    log_info(f"Save RL Model: {SAVE_RL_MODEL}", module_name)
    log_info(f"Load RL Model: {LOAD_RL_MODEL}", module_name)
    log_info(f"Use New Model: {USE_NEW_MODEL}", module_name)
    log_info(f"Model Save Frequency: {MODEL_SAVE_FREQUENCY}", module_name)
    log_info(f"Auto Save Final: {AUTO_SAVE_FINAL}", module_name)

    # --- Checkpoint Setup ---
    if checkpoint_base_dir is None:
        script_dir = Path(__file__).parent
        # Adjust path relative to benchmark.py if needed
        project_root_fallback = script_dir.parents[1]  # Assuming Benchmark is one level down
        checkpoint_base_dir = project_root_fallback / "SAPSO" / "checkpoints"  # Example adjusted path
        log_warning(f"checkpoint_base_dir not provided, using default: {checkpoint_base_dir}", module_name)

    # Add 'train' subdirectory for all training outputs
    train_output_dir = Path(checkpoint_base_dir) / "train"
    os.makedirs(train_output_dir, exist_ok=True)

    # Generate model filename based on CONFIG settings
    mode_suffix = "adaptive_nt" if adaptive_nt_mode else f"fixed_nt{agent_step_size}"
    timestamp_suffix = f"_{int(time.time())}" if INCLUDE_TIMESTAMP else ""
    version_suffix = MODEL_VERSION_SUFFIX if MODEL_VERSION_SUFFIX else ""
    
    model_filename = f"{MODEL_NAME_PREFIX}_{mode_suffix}{version_suffix}{timestamp_suffix}"
    checkpoint_dir = train_output_dir / f"checkpoints_sapso_vectorized_{mode_suffix}"
    checkpoint_file = checkpoint_dir / f"{model_filename}_checkpoint.pth"
    final_model_file = checkpoint_dir / f"{model_filename}_final.pth"
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_info(f"Checkpoints will be saved in: {checkpoint_dir}", module_name)
    log_info(f"Model filename: {model_filename}", module_name)

    # --- Load Existing Model if Configured ---
    if LOAD_RL_MODEL:
        # Try to find and load the most recent model
        model_files = list(checkpoint_dir.glob(f"{MODEL_NAME_PREFIX}_{mode_suffix}*_final.pth"))
        if model_files:
            # Sort by modification time and get the most recent
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            try:
                agent.load(str(latest_model))
                log_success(f"Loaded existing model from: {latest_model}", module_name)
            except Exception as e:
                log_error(f"Failed to load existing model: {e}", module_name)
                if not USE_NEW_MODEL:
                    log_error("Cannot continue without loading model. Exiting.", module_name)
                    return
        else:
            log_warning(f"No existing model found in {checkpoint_dir}", module_name)
            if not USE_NEW_MODEL:
                log_error("Cannot continue without loading model. Exiting.", module_name)
                return
    elif not USE_NEW_MODEL:
        log_warning("USE_NEW_MODEL is False but LOAD_RL_MODEL is False. Using new model.", module_name)

    # --- Initialize Replay Buffer ---
    buffer_capacity = 1_000_000
    buffer = ReplayBuffer(
        capacity=buffer_capacity,
        state_dim=state_dim,
        action_dim=action_dim,
        device=device
    )

    # --- Tracking Variables ---
    results_log = {}  # {func_name: [(ep1_reward, ep1_gbest), ...]}
    global_step_count = 0
    total_agent_steps = 0
    total_episodes_run = 0
    current_run_id = 0  # Track run ID for metrics
    
    # --- Metrics Collection for Plotting ---
    metrics_collector = None  # Will be initialized with the first environment's metrics calculator
    accumulated_metrics = {}  # Store metrics data across all episodes

    # --- Main Training Loop (Nested) ---
    log_header("Starting training...", module_name)
    train_start_time = time.time()

    # Outer loop: Iterate through each objective function
    random.shuffle(objective_function_classes)
    # for func_index, selected_func_class in enumerate(objective_function_classes):
    for func_index, selected_func_class in enumerate(objective_function_classes[:5]):
        func_name = selected_func_class.__name__
        log_header(f"===== Training on Function {func_index + 1}/{len(objective_function_classes)}: {func_name} =====",
                   module_name)
        results_log[func_name] = []

        # Inner loop: Run N episodes for the current function
        for episode_num in range(episodes_per_function):
            total_episodes_run += 1
            current_run_id += 1  # Increment run ID for each episode
            log_info(
                f"--- Episode {episode_num + 1}/{episodes_per_function} (Total Ep: {total_episodes_run}) | Function: {func_name} | Run ID: {current_run_id} ---",
                module_name)

            train_env = None
            try:
                current_dim = env_dim
                if func_name == "GiuntaFunction":
                    current_dim = 2
                    log_info(f"  Adjusting dimension to {current_dim} for {func_name}", module_name)

                # Objective function doesn't need num_particles directly
                obj_func_instance = selected_func_class(dim=current_dim)

                # Store the run ID for this episode to pass to the environment
                episode_run_id = current_run_id
                
                # --- Use PSOEnvVectorized ---
                train_env = Environment(
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
                    # stability_threshold=stability_threshold
                    run_id=episode_run_id
                )
                
                # Initialize metrics collector with the first environment's metrics calculator
                if metrics_collector is None and hasattr(train_env, 'pso') and hasattr(train_env.pso, 'metrics_calculator'):
                    metrics_collector = train_env.pso.metrics_calculator
                    log_info("Initialized metrics collector for plotting", module_name)
                
                # Use total_episodes_run for seed consistency across restarts if needed
                # TODO eval seed effect
                # state, _ = train_env.reset(seed=total_episodes_run)
                state, _ = train_env.reset()

            except Exception as e:
                log_error(f"Error creating environment for {func_name}, episode {episode_num + 1}: {e}", module_name)
                log_error(traceback.format_exc(), module_name)
                log_warning("Skipping this episode.", module_name)
                continue  # Skip to the next episode

            episode_reward = 0.0
            terminated, truncated = False, False
            episode_agent_steps = 0
            # Track the best gbest found *during* this episode
            episode_best_gbest = float(train_env.pso.gbest_value) if train_env and hasattr(train_env, 'pso') else float('inf')

            # --- Innermost Loop (Agent Steps within Episode) ---
            while not terminated and not truncated:
                if total_agent_steps < start_steps:
                    action = train_env.action_space.sample()
                else:
                    # Ensure state is numpy array for agent
                    if isinstance(state, torch.Tensor):
                        state_np = state.cpu().numpy()
                    else:
                        state_np = np.array(state, dtype=np.float32)
                    action = agent.select_action(state_np)

                try:
                    # The Environment now automatically passes the function name and run ID to the PSO
                    # The metrics tracking is handled internally by the Environment and PSOSwarm
                    next_state, reward, terminated, truncated, info = train_env.step(action)
                    # Update episode's best gbest using the final gbest from the agent turn
                    turn_final_gbest = info.get('final_gbest', np.inf)
                    if np.isfinite(turn_final_gbest) and isinstance(turn_final_gbest, (int, float)):
                        episode_best_gbest = min(episode_best_gbest, float(turn_final_gbest))

                except Exception as e:
                    log_error(
                        f"Error during env.step() in episode {episode_num + 1}, agent step {episode_agent_steps + 1}: {e}",
                        module_name)
                    log_error(traceback.format_exc(), module_name)
                    log_warning("Terminating episode early.", module_name)
                    truncated = True  # Mark as truncated due to error
                    next_state = state  # Use previous state
                    reward = 0  # Assign no reward for error step
                    info = {}  # Empty info

                # Update counters
                episode_agent_steps += 1
                total_agent_steps += 1
                pso_steps_this_turn = info.get('steps_taken', 0)
                current_nt_used = info.get('nt', agent_step_size if not adaptive_nt_mode else nt_range[0])
                global_step_count += pso_steps_this_turn

                # Store experience
                done_flag = terminated or truncated
                # Ensure state arrays are float32 for buffer
                buffer.push(np.array(state, dtype=np.float32),
                            np.array(action, dtype=np.float32),
                            float(reward),
                            np.array(next_state, dtype=np.float32),
                            bool(done_flag))

                state = next_state
                episode_reward += reward

                # Agent learning update
                if len(buffer) >= batch_size and total_agent_steps >= start_steps:
                    for _ in range(updates_per_step):
                        agent.update(buffer, batch_size)

                # Print progress (less frequent within episode loop)
                # print_freq_train = 200 # Can uncomment for more verbose logging
                # if episode_agent_steps % print_freq_train == 0:
                #      log_debug(f"    Ep {episode_num+1} | Agt Step {episode_agent_steps} "
                #                f"| PSO Step {train_env.current_step}/{train_env.max_steps} "
                #                f"| Buf {len(buffer)} | R: {reward:.4f} (nt={current_nt_used})", module_name)

            # --- End of Episode ---
            # Final gbest for the episode is the best one tracked
            final_gbest_for_log = episode_best_gbest
            # Fallback: if tracking failed, get final value from pso object
            if not np.isfinite(final_gbest_for_log) and train_env and hasattr(train_env, 'pso'):
                final_gbest_for_log = train_env.pso.gbest_value

            results_log[func_name].append((episode_reward, final_gbest_for_log))

            log_info(f"--- Episode {episode_num + 1} Finished ---", module_name)
            log_info(f"  Agent Steps: {episode_agent_steps}", module_name)
            log_info(f"  Total Reward: {episode_reward:.4f}", module_name)
            log_info(f"  Final GBest: {final_gbest_for_log:.6e}", module_name)
            log_info(f"  Global PSO Steps: {global_step_count}", module_name)
            log_info(f"  Total Agent Steps: {total_agent_steps}", module_name)

            # Store metrics data from this episode for plotting
            if train_env and hasattr(train_env, 'pso') and hasattr(train_env.pso, 'metrics_calculator'):
                episode_metrics = train_env.pso.metrics_calculator
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
                if train_env: train_env.close()
            except Exception as e:
                log_warning(f"Error closing environment for episode {episode_num + 1}: {e}", module_name)

        # --- End of Episodes for Current Function ---
        log_header(f"===== Function {func_name} Training Complete =====", module_name)
        if results_log[func_name]:
            # Filter out potential non-finite rewards if errors occurred
            finite_rewards = [r for r, g in results_log[func_name] if np.isfinite(r)]
            func_gbests = [g for r, g in results_log[func_name] if np.isfinite(g)]
            if finite_rewards:
                log_info(f"  Avg Reward over {len(finite_rewards)} valid episodes: {np.mean(finite_rewards):.4f}",
                         module_name)
            else:
                log_warning(f"  No finite reward values recorded for this function.", module_name)

            if func_gbests:
                log_info(f"  Avg Final GBest over {len(func_gbests)} valid episodes: {np.mean(func_gbests):.6e}",
                         module_name)
            else:
                log_warning(f"  No finite Final GBest values recorded for this function.", module_name)
        else:
            log_warning(f"  No results logged for this function.", module_name)

        # --- Conditional Checkpoint Saving ---
        if SAVE_RL_MODEL and (func_index + 1) % MODEL_SAVE_FREQUENCY == 0:
            try:
                agent.save(str(checkpoint_file))
                log_success(f"Checkpoint saved after function {func_index + 1} ({func_name}) to {checkpoint_file}",
                            module_name)
            except Exception as e:
                log_warning(f"Could not save checkpoint during training: {e}", module_name)

    # --- End of Training Loop (All Functions) ---
    train_end_time = time.time()
    log_header(f"Training finished. Total time: {train_end_time - train_start_time:.2f} seconds.", module_name)
    log_info(f"Total Episodes Run: {total_episodes_run}", module_name)

    # --- Calculate and Print Overall Final Statistics ---
    all_final_rewards = []
    all_final_gbests = []
    for func_name, results in results_log.items():
        all_final_rewards.extend([r for r, g in results if np.isfinite(r)])  # Only use finite rewards
        all_final_gbests.extend([g for r, g in results if np.isfinite(g)])  # Only use finite gbests

    log_header("--- Overall Training Results ---", module_name)
    if all_final_rewards:
        mean_reward = np.mean(all_final_rewards)
        std_reward = np.std(all_final_rewards)
        var_reward = np.var(all_final_rewards)
        log_info(
            f"Overall Mean Training Reward ({len(all_final_rewards)} valid Eps): {mean_reward:.4f} (μ) +/- {std_reward:.4f} (σ)",
            module_name)
        log_info(f"Overall Training Reward Variance: {var_reward:.6f}", module_name)
    else:
        log_warning("No valid training reward data collected.", module_name)

    # --- Calculate NORMALIZED GBest Statistics ---
    if all_final_gbests:
        # Convert to numpy array for easier manipulation
        gbest_array = np.array(all_final_gbests)

        # Calculate observed min and max
        min_gbest = np.min(gbest_array)
        max_gbest = np.max(gbest_array)
        gbest_range = max_gbest - min_gbest

        log_info(f"Observed GBest Range: [{min_gbest:.6e}, {max_gbest:.6e}]", module_name)

        if gbest_range > 1e-12:  # Check if range is significantly greater than zero
            # Normalize gbests to [0, 1] based on observed range
            normalized_gbests = (gbest_array - min_gbest) / gbest_range

            # Calculate mean and std dev of normalized values
            mean_normalized_gbest = np.mean(normalized_gbests)
            std_normalized_gbest = np.std(normalized_gbests)
            var_normalized_gbest = np.var(normalized_gbests)  # Variance of normalized

            log_info(
                f"Overall Mean NORMALIZED Final GBest ({len(all_final_gbests)} valid Eps): {mean_normalized_gbest:.6f} (μ) +/- {std_normalized_gbest:.6f} (σ)",
                module_name)
            log_info(f"Overall NORMALIZED Final GBest Variance: {var_normalized_gbest:.6f}", module_name)

            # Optionally, print raw mean/std as well for comparison
            mean_raw_gbest = np.mean(gbest_array)
            std_raw_gbest = np.std(gbest_array)
            log_info(f"(Raw Mean Final GBest: {mean_raw_gbest:.6e} +/- {std_raw_gbest:.6e})", module_name)

        else:
            # Handle case where all gbest values are identical (or very close)
            log_warning(f"All valid final GBest values are nearly identical ({min_gbest:.6e}). Normalization skipped.",
                        module_name)
            log_info(f"Raw Mean Final GBest: {min_gbest:.6e} +/- 0.0", module_name)

    else:
        log_warning("No final global best data collected for normalization.", module_name)
    log_info("---------------------------------", module_name)

    # --- Conditional Final Model Save ---
    if SAVE_RL_MODEL and AUTO_SAVE_FINAL:
        try:
            agent.save(str(final_model_file))
            log_success(f"Final trained model saved to {final_model_file}", module_name)
        except Exception as e:
            log_warning(f"Could not save final checkpoint: {e}", module_name)
    elif not SAVE_RL_MODEL:
        log_info("Model saving disabled. Final model not saved.", module_name)

    # --- Plot Training Reward Curve (Average across functions per 'epoch') ---
    avg_rewards_per_func = {}
    for func_name, results in results_log.items():
        # Calculate average only from finite rewards for that function
        finite_rewards_func = [r for r, g in results if np.isfinite(r)]
        if finite_rewards_func:
            avg_rewards_per_func[func_name] = np.mean(finite_rewards_func)
        # else: # Optionally handle functions with no valid rewards
        #     avg_rewards_per_func[func_name] = np.nan

    if avg_rewards_per_func:
        plt.figure(figsize=(12, 6))
        # Filter out potential NaN values if some functions had no valid rewards
        valid_func_names = [name for name, avg in avg_rewards_per_func.items() if np.isfinite(avg)]
        valid_avg_rewards = [avg for avg in avg_rewards_per_func.values() if np.isfinite(avg)]

        if valid_func_names:
            plt.bar(valid_func_names, valid_avg_rewards)
            plt.title(f"SAC Avg Reward per Function ({episodes_per_function} eps each) {mode_suffix}")
            plt.xlabel("Objective Function")
            plt.ylabel("Average Episode Reward")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save the plot in the train directory
            timestamped_filename = generate_timestamped_filename(f"{model_filename}_train_rewards_per_func_static")
            plot_path = train_output_dir / timestamped_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            log_success(f"Training reward plot saved to: {plot_path}", module_name)
        else:
            log_warning("No valid function names for plotting.", module_name)
    else:
        log_warning("No average rewards calculated for plotting.", module_name)

    # --- Generate SAPSO Metric Plots ---
    log_header("Generating SAPSO metric plots", module_name)
    
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
                plotter = SAPSOPlotter(str(train_output_dir), plot_only_averages=PLOT_ONLY_AVERAGES)
                plotter.plot_all_metrics(mock_metrics, save_plots=True, show_plots=False)
                log_success("SAPSO metric plots generated successfully", module_name)
            else:
                log_warning("No valid metrics or parameter data found in accumulated data", module_name)
        except Exception as e:
            log_error(f"Error generating SAPSO metric plots: {e}", module_name)
            log_error(traceback.format_exc(), module_name)
    else:
        log_warning("No metrics data available for plotting. Metrics collection may not be properly configured.", module_name)
        log_info("To enable metrics plotting, ensure the PSO environment is properly configured with metrics tracking.", module_name)

    return agent, results_log

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train SAPSO agent with enhanced metrics tracking")
    
    # Environment parameters
    parser.add_argument("--env-dim", type=int, default=ENV_DIM, help="Environment dimension")
    parser.add_argument("--env-particles", type=int, default=ENV_PARTICLES, help="Number of particles")
    parser.add_argument("--env-max-steps", type=int, default=ENV_MAX_STEPS, help="Maximum steps per episode")
    parser.add_argument("--agent-step-size", type=int, default=AGENT_STEP_SIZE, help="Agent step size")
    parser.add_argument("--adaptive-nt-mode", type=str, default=str(ADAPTIVE_NT_MODE), help="Adaptive NT mode (true/false)")
    parser.add_argument("--nt-range", type=str, default=str(NT_RANGE), help="NT range as string '[min, max]'")
    parser.add_argument("--episodes-per-function", type=int, default=EPISODES_PER_FUNCTION, help="Episodes per function")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--start-steps", type=int, default=START_STEPS, help="Start steps")
    parser.add_argument("--updates-per-step", type=int, default=UPDATES_PER_STEP, help="Updates per step")
    parser.add_argument("--save-freq-multiplier", type=int, default=SAVE_FREQ_MULTIPLIER, help="Save frequency multiplier")
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
    
    # Run training
    train_agent(
        env_dim=args.env_dim,
        env_particles=args.env_particles,
        env_max_steps=args.env_max_steps,
        agent_step_size=args.agent_step_size,
        adaptive_nt_mode=adaptive_nt_mode,
        nt_range=nt_range,
        episodes_per_function=args.episodes_per_function,
        batch_size=args.batch_size,
        start_steps=args.start_steps,
        updates_per_step=args.updates_per_step,
        save_freq_multiplier=args.save_freq_multiplier,
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
