import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import collections # Import collections for defaultdict

# Assuming these imports are correctly set up based on your project structure
from LLM.PSO.ObjectiveFunctions.Rastrgin import RastriginFunction
from LLM.RL.ActorCritic.Agent import SACAgent
from LLM.RL.Replay.ReplayBuffer import ReplayBuffer
from LLM.SAPSO.Gyms.PSO_GymV2 import PSOEnv # Use the modified PSOEnv

# === Import the plotting functions from the new file ===
# Assumes graphing.py contains all the necessary plotting functions now
# Ensure the path 'LLM.SAPSO.graphics.graphing' is correct for your project structure
from LLM.SAPSO.graphics.graphing import (
    plot_evaluation_parameters,
    plot_stable_particles,      # Add new imports
    plot_infeasible_particles,
    plot_average_velocity,
    plot_swarm_diversity,
    plot_gbest_convergence      # Add gbest plot
)
# ======================================================

# --- Main Function ---
def main():
    """Main function to train the SAC agent for PSO control."""
    # === Setup ===
    env_dim = 30
    env_particles = 30
    env_max_steps = 5000 # Total PSO steps allowed per episode
    agent_step_size = 125 # Used for fixed Nt mode

    # --- Configuration: Toggle for adaptive nt ---
    ADAPTIVE_NT_MODE = False # Set to True/False
    NT_RANGE = (1, 50)       # Range if ADAPTIVE_NT_MODE is True

    # Create the objective function instance
    obj_func = RastriginFunction(env_dim, env_particles)

    # --- Initialize Environment ---
    # Use a factory function or lambda to easily create env instances
    def create_env():
        # Ensure PSO_GymV2 is updated to return detailed step metrics in info dict
        return PSOEnv(
            obj_func=obj_func,
            max_steps=env_max_steps,
            agent_step_size=agent_step_size, # Pass fixed step size
            adaptive_nt=ADAPTIVE_NT_MODE,
            nt_range=NT_RANGE
        )

    # Create a sample env to get dimensions
    temp_env = create_env()
    state_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.shape[0] # Will be 3 or 4
    temp_env.close() # Close the temporary environment

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Configuration ---")
    print(f"Using device: {device}")
    print(f"Adaptive Nt Mode: {ADAPTIVE_NT_MODE}")
    if not ADAPTIVE_NT_MODE:
        print(f"Fixed Agent Step Size (Nt): {agent_step_size}")
    else:
        print(f"Adaptive Nt Range: {NT_RANGE}")
    print(f"State Dimension: {state_dim}")
    print(f"Action Dimension: {action_dim}")
    print(f"--------------------")

    # --- Initialize Agent ---
    # Agent Hyperparameters
    hidden_dim=256; gamma=1.0; tau=0.005; alpha=0.2; actor_lr=3e-4; critic_lr=3e-4
    agent = SACAgent(
        state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim,
        gamma=gamma, tau=tau, alpha=alpha, actor_lr=actor_lr, critic_lr=critic_lr,
        device=device, adaptive_nt=ADAPTIVE_NT_MODE
    )

    # --- Initialize Replay Buffer ---
    buffer_capacity = 1_000_000
    buffer = ReplayBuffer(
        capacity=buffer_capacity, state_dim=state_dim, action_dim=action_dim, device=device
    )

    # === Training Hyperparameters ===
    num_episodes = 10 # Number of training episodes
    batch_size = 256
    start_steps = 1000 # Agent steps before learning starts
    updates_per_step = 1 # Agent updates per agent step
    print_freq = 10 # Print frequency during training (agent steps)

    # --- Tracking Variables ---
    rewards_per_episode = []
    global_step_count = 0 # Total PSO steps across training
    total_agent_steps = 0 # Total agent steps across training

    # --- Checkpoint Loading ---
    checkpoint_dir = "checkpoints_sapso"
    # Checkpoint name includes fixed nt value if not adaptive
    checkpoint_prefix = "sac_psoenv_adaptive_nt" if ADAPTIVE_NT_MODE else f"sac_psoenv_fixed_nt{agent_step_size}"
    checkpoint_file = os.path.join(checkpoint_dir, f"{checkpoint_prefix}_checkpoint.pth")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Option to load checkpoint or force retraining
    load_checkpoint = True # Set to False to always retrain
    training_needed = True # Flag to control if training loop runs

    if load_checkpoint and os.path.exists(checkpoint_file):
        try:
            agent.load(checkpoint_file)
            print(f"Loaded checkpoint from {checkpoint_file}")
            training_needed = False # Skip training if loaded successfully
        except Exception as e:
            print(f"Could not load checkpoint: {e}. Starting training from scratch.")
            load_checkpoint = False # Ensure training happens if load fails
    elif load_checkpoint:
        print("No checkpoint found. Starting training from scratch.")
        load_checkpoint = False
    else:
         print("Starting training from scratch (load_checkpoint is False).")

    if num_episodes <= 0:
         training_needed = False # Don't train if num_episodes is zero


    # --- Main Training Loop ---
    if training_needed:
        print("Starting training...")
        train_start_time = time.time()
        train_env = create_env() # Use a dedicated env instance for training

        for episode in range(num_episodes):
            state, _ = train_env.reset(seed=episode) # Seed for reproducibility
            episode_reward = 0.0
            terminated, truncated = False, False
            episode_agent_steps = 0

            while not terminated and not truncated:
                # Select action (random exploration initially)
                if total_agent_steps < start_steps:
                    action = train_env.action_space.sample()
                else:
                    action = agent.select_action(state)

                # Environment step
                next_state, reward, terminated, truncated, info = train_env.step(action)

                # Update counters
                episode_agent_steps += 1
                total_agent_steps += 1
                pso_steps_this_turn = info.get('steps_taken', 1)
                current_nt_used = info.get('nt', agent_step_size if not ADAPTIVE_NT_MODE else 1)
                global_step_count += pso_steps_this_turn

                # Store experience
                done_flag = terminated or truncated
                buffer.push(state, action, reward, next_state, done_flag)

                # Update state
                state = next_state
                episode_reward += reward

                # Agent learning update
                if len(buffer) >= batch_size and total_agent_steps >= start_steps:
                    for _ in range(updates_per_step):
                        agent.update(buffer, batch_size)

                # Print progress
                if episode_agent_steps % print_freq == 0:
                     print(f"  Train Ep {episode+1}/{num_episodes} | Agt Step {episode_agent_steps} "
                           f"| PSO Step {train_env.current_step}/{train_env.max_steps} "
                           f"| Turn Reward: {reward:.4f} (nt={current_nt_used})")

                # Check step limit
                if train_env.current_step >= train_env.max_steps:
                    truncated = True

            # End of episode
            rewards_per_episode.append(episode_reward)
            print(f"Train Episode {episode+1}/{num_episodes} | Agent Steps: {episode_agent_steps} "
                  f"| Total Reward: {episode_reward:.4f} | Global PSO Steps: {global_step_count}")

            # Periodic checkpoint saving
            if (episode + 1) % 10 == 0:
                try:
                    agent.save(checkpoint_file)
                except Exception as e:
                    print(f"Warning: Could not save checkpoint during training: {e}")

        # End of training loop
        train_end_time = time.time()
        print(f"Training finished. Total time: {train_end_time - train_start_time:.2f} seconds.")
        train_env.close() # Close the training environment

        # --- Save Final Model After Training ---
        final_checkpoint_file = checkpoint_file.replace("_checkpoint.pth", "_final.pth")
        try:
            agent.save(final_checkpoint_file)
            print(f"Final trained model saved to {final_checkpoint_file}")
        except Exception as e:
            print(f"Warning: Could not save final checkpoint: {e}")

        # --- Plot Training Reward Curve ---
        if rewards_per_episode:
            plt.figure(figsize=(10, 5))
            plt.plot(rewards_per_episode)
            mode_str = '(Adaptive Nt)' if ADAPTIVE_NT_MODE else f'(Fixed Nt={agent_step_size})'
            plt.title(f"SAC Training Rewards {mode_str}")
            plt.xlabel("Episode")
            plt.ylabel("Total Reward per Episode")
            plt.grid(True)
            plot_filename = os.path.join(checkpoint_dir, f"{checkpoint_prefix}_train_rewards.png")
            try:
                plt.savefig(plot_filename)
                print(f"Training reward plot saved to {plot_filename}")
            except Exception as e:
                print(f"Warning: Could not save training reward plot: {e}")
            plt.show() # Show plot after saving

    elif not training_needed and load_checkpoint:
         print("Skipping training as checkpoint was loaded.")
    else:
         print("Skipping training as num_episodes is 0 or loading failed.")


    # === Multi-Run Evaluation for Plotting ===
    num_eval_runs = 30 # Number of deterministic runs for statistics
    print(f"\nStarting {num_eval_runs} deterministic evaluation runs...")

    # Data structure: {pso_step: [data_point_run1, data_point_run2, ...], ...}
    # data_point = [w, c1, c2, stable%, infeasible%, avg_vel_mag, diversity, gbest]
    evaluation_data = collections.defaultdict(list)
    all_eval_rewards = []
    eval_start_time = time.time()

    for run in range(num_eval_runs):
        print(f"  Starting Evaluation Run {run + 1}/{num_eval_runs}...")
        eval_env = create_env() # Create a fresh env for each run
        # Use different seeds for evaluation runs
        state, _ = eval_env.reset(seed=num_episodes + run + 100)
        run_reward = 0.0
        term, trunc = False, False
        # current_pso_step = 0 # Track based on env.current_step

        while not term and not trunc:
            # Select action deterministically
            action = agent.select_action(state, deterministic=True)

            # Get the CPs that *will be used* for the next nt steps
            eval_omega, eval_c1, eval_c2, eval_nt = eval_env._rescale_action(action)
            # Store CPs once per agent turn, will be added to each step's data
            params_for_turn = [float(eval_omega), float(eval_c1), float(eval_c2)]

            # Step the environment - info dict should contain 'step_metrics' list
            next_state, reward, term, trunc, info = eval_env.step(action)

            # --- Collect detailed metrics per PSO step ---
            step_metrics_list = info.get('step_metrics', []) # Get list of dicts

            for step_metric_dict in step_metrics_list:
                pso_step_index = step_metric_dict.get('pso_step', -1)
                if pso_step_index >= 0 and pso_step_index < eval_env.max_steps:
                    # Extract metrics for this specific PSO step
                    stable_ratio = step_metric_dict.get('stable_ratio', np.nan)
                    # Note: feasible_ratio is calculated, infeasible = 1 - feasible
                    feasible_ratio = step_metric_dict.get('feasible_ratio', np.nan)
                    infeasible_ratio = 1.0 - feasible_ratio if not np.isnan(feasible_ratio) else np.nan
                    # Use the correct key for velocity magnitude
                    avg_vel_mag = step_metric_dict.get('avg_velocity_magnitude', step_metric_dict.get('avg_velocity', np.nan))
                    diversity = step_metric_dict.get('diversity', np.nan)
                    gbest_val = step_metric_dict.get('gbest_value', np.nan) # gbest at this step

                    # Store all metrics for this step in the defined order
                    data_point = params_for_turn + [
                        stable_ratio, infeasible_ratio, avg_vel_mag, diversity, gbest_val
                    ]
                    evaluation_data[pso_step_index].append(data_point)
                # else: print(f"Debug: Invalid pso_step_index {pso_step_index}") # Optional debug

            # Update state and reward
            state = next_state
            run_reward += reward

            # Check step limit based on environment's internal counter
            if eval_env.current_step >= eval_env.max_steps:
                trunc = True

        # End of evaluation episode run
        all_eval_rewards.append(run_reward)
        print(f"  Finished Run {run + 1}/{num_eval_runs}. Reward: {run_reward:.4f}")
        eval_env.close() # Close the environment for this run

    # End of all evaluation runs
    eval_end_time = time.time()
    print(f"Finished {num_eval_runs} evaluation runs. Total time: {eval_end_time - eval_start_time:.2f} seconds.")
    if all_eval_rewards:
        print(f"Mean Evaluation Reward: {np.mean(all_eval_rewards):.4f} +/- {np.std(all_eval_rewards):.4f}")

    # --- Generate ALL Evaluation Plots ---
    if evaluation_data:
        print("\nGenerating evaluation plots...")
        # Call the plotting functions imported from graphing.py
        plot_evaluation_parameters(evaluation_data, env_max_steps, checkpoint_dir, checkpoint_prefix)
        plot_stable_particles(evaluation_data, env_max_steps, checkpoint_dir, checkpoint_prefix)
        plot_infeasible_particles(evaluation_data, env_max_steps, checkpoint_dir, checkpoint_prefix)
        plot_average_velocity(evaluation_data, env_max_steps, checkpoint_dir, checkpoint_prefix)
        plot_swarm_diversity(evaluation_data, env_max_steps, checkpoint_dir, checkpoint_prefix)
        plot_gbest_convergence(evaluation_data, env_max_steps, checkpoint_dir, checkpoint_prefix) # Default uses log scale now
    else:
        print("No evaluation data was collected, skipping plot generation.")


if __name__ == "__main__":
    # Ensure the script runs when executed directly
    main()
