# train.py - Place in PSO-ToyBox/SAPSO_AGENT/Benchmark/
# Uses static imports (absolute paths from SAPSO_AGENT) for objective functions
# Accepts hyperparameters as arguments
# Trains N episodes for EACH function sequentially
# Calculates and prints final training reward and NORMALIZED gbest mean/std
# --- UPDATED TO USE PSOEnvVectorized ---
# --- UPDATED to move all imports to top ---
# --- UPDATED to normalize final gbest statistics ---
# --- UPDATED to use logger ---

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import traceback  # For logging exceptions
from pathlib import Path
from SAPSO_AGENT.SAPSO.RL.ActorCritic.Agent import SACAgent
from SAPSO_AGENT.SAPSO.RL.Replay.ReplayBuffer import ReplayBuffer
from SAPSO_AGENT.SAPSO.Environment.Environment import Environment

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.Training.Loader import objective_function_classes
from SAPSO_AGENT.Logs.logger import *

# --- Main Training Function (Accepts Arguments) ---
def train_agent(
        env_dim=30,
        env_particles=30,  # Keep this parameter
        env_max_steps=5000,
        agent_step_size=125,
        adaptive_nt_mode=False,
        nt_range=(1, 50),
        episodes_per_function=5,
        batch_size=256,
        start_steps=1000,
        updates_per_step=1,
        save_freq_multiplier=4,
        checkpoint_base_dir=None,
        # Agent HPs
        hidden_dim=256,
        gamma=1.0,
        tau=0.005,
        alpha=0.2,
        actor_lr=3e-4,
        critic_lr=3e-4,
        # Add PSO/Env params if needed by PSOEnvVectorized constructor
        v_clamp_ratio=0.2,
        use_velocity_clamping=True,
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
        state_dim = temp_env.observation_space.shape[0]
        action_dim = temp_env.action_space.shape[0]
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
    log_info(f"Env Dim: {env_dim}, Particles: {env_particles}, Max PSO Steps: {env_max_steps}", module_name)
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

    # --- Checkpoint Setup ---
    if checkpoint_base_dir is None:
        script_dir = Path(__file__).parent
        # Adjust path relative to benchmark.py if needed
        project_root_fallback = script_dir.parents[1]  # Assuming Benchmark is one level down
        checkpoint_base_dir = project_root_fallback / "SAPSO" / "checkpoints"  # Example adjusted path
        log_warning(f"checkpoint_base_dir not provided, using default: {checkpoint_base_dir}", module_name)

    # Adjusted checkpoint naming convention slightly
    mode_suffix = "adaptive_nt" if adaptive_nt_mode else f"fixed_nt{agent_step_size}"
    checkpoint_dir = Path(checkpoint_base_dir) / f"checkpoints_sapso_vectorized_{mode_suffix}"
    checkpoint_prefix = f"sac_psoenv_vectorized_{mode_suffix}"  # Prefix used for files
    checkpoint_file = checkpoint_dir / f"{checkpoint_prefix}_checkpoint.pth"
    final_model_file = checkpoint_dir / f"{checkpoint_prefix}_final.pth"
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_info(f"Checkpoints will be saved in: {checkpoint_dir}", module_name)

    # --- Main Training Loop (Nested) ---
    log_header("Starting training...", module_name)
    train_start_time = time.time()

    # Outer loop: Iterate through each objective function
    for func_index, selected_func_class in enumerate(objective_function_classes):
        func_name = selected_func_class.__name__
        log_header(f"===== Training on Function {func_index + 1}/{len(objective_function_classes)}: {func_name} =====",
                   module_name)
        results_log[func_name] = []

        # Inner loop: Run N episodes for the current function
        for episode_num in range(episodes_per_function):
            total_episodes_run += 1
            log_info(
                f"--- Episode {episode_num + 1}/{episodes_per_function} (Total Ep: {total_episodes_run}) | Function: {func_name} ---",
                module_name)

            train_env = None
            try:
                current_dim = env_dim
                if func_name == "GiuntaFunction":
                    current_dim = 2
                    log_info(f"  Adjusting dimension to {current_dim} for {func_name}", module_name)

                # Objective function doesn't need num_particles directly
                obj_func_instance = selected_func_class(dim=current_dim)

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
                )
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
            episode_best_gbest = train_env.pso.gbest_value if train_env and hasattr(train_env, 'pso') else np.inf

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
                    next_state, reward, terminated, truncated, info = train_env.step(action)
                    # Update episode's best gbest using the final gbest from the agent turn
                    turn_final_gbest = info.get('final_gbest', np.inf)
                    if np.isfinite(turn_final_gbest):
                        episode_best_gbest = min(episode_best_gbest, turn_final_gbest)

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

        # --- Periodic Checkpoint Saving (Based on Functions Completed) ---
        # Save checkpoint every 'save_freq_multiplier' *functions*
        if (func_index + 1) % save_freq_multiplier == 0:
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

    # --- Save Final Model ---
    try:
        agent.save(str(final_model_file))
        log_success(f"Final trained model saved to {final_model_file}", module_name)
    except Exception as e:
        log_warning(f"Could not save final checkpoint: {e}", module_name)

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
            plt.ylabel(f"Average Reward over Episodes")
            plt.xticks(rotation=90)
            plt.grid(axis='y')
            plt.tight_layout()
            plot_filename = checkpoint_dir / f"{checkpoint_prefix}_train_rewards_per_func_static.png"
            try:
                plt.savefig(str(plot_filename))
                log_success(f"Training reward plot (per function) saved to {plot_filename}", module_name)
            except Exception as e:
                log_warning(f"Could not save training reward plot: {e}", module_name)
            # plt.show() # Optionally show plot immediately
            plt.close()  # Close the figure to free memory
        else:
            log_warning("No valid average rewards per function to plot.", module_name)
    else:
        log_warning("No results logged for plotting average rewards per function.", module_name)

# --- Main execution block removed ---
# This script is intended to be called by benchmark.py
