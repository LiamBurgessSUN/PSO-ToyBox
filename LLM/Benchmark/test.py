# test.py - Place in PSO-ToyBox/LLM/SAPSO/mains/
# Uses static imports and accepts hyperparameters as arguments
# Calculates and prints final gbest mean/std

import torch
import numpy as np
import os
import time
import sys
import collections
from pathlib import Path

# --- Project Imports ---
try:
    from LLM.RL.ActorCritic.Agent import SACAgent
    from LLM.SAPSO.Gyms.PSO_GymV2 import PSOEnv
    from LLM.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction # Base class needed
    # Import plotting functions
    from LLM.SAPSO.graphics.graphing import (
        plot_evaluation_parameters,
        plot_stable_particles,
        plot_infeasible_particles,
        plot_average_velocity,
        plot_swarm_diversity,
        plot_gbest_convergence
    )
    # --- Static Imports for Testing Objective Functions ---
    from LLM.PSO.ObjectiveFunctions.Testing.CrossLegTable import CrossLegTableFunction
    from LLM.PSO.ObjectiveFunctions.Testing.Lanczos import Lanczos3Function
    from LLM.PSO.ObjectiveFunctions.Testing.Michalewicz import MichalewiczFunction
    from LLM.PSO.ObjectiveFunctions.Testing.Schaffer import Schaffer4Function
    from LLM.PSO.ObjectiveFunctions.Testing.SineEnvelope import SineEnvelopeFunction
    from LLM.PSO.ObjectiveFunctions.Testing.StretchedVSineWave import StretchedVSineWaveFunction
    from LLM.PSO.ObjectiveFunctions.Testing.Wavy import WavyFunction

except ImportError as e:
    print(f"Error importing project modules in test.py: {e}")
    print("Please ensure the script is run from a directory where the 'LLM' package is discoverable,")
    print("or that the project root directory (containing LLM) is in your PYTHONPATH.")
    sys.exit(1)
# --- End Project Imports ---


# --- Manually Create List of Test Objective Function Classes ---
test_objective_function_classes = [
    CrossLegTableFunction, Lanczos3Function, MichalewiczFunction, Schaffer4Function,
    SineEnvelopeFunction, StretchedVSineWaveFunction, WavyFunction
]
# --- End Manual List ---


# --- Main Testing Function (Accepts Arguments) ---
def test_agent(
    env_dim=30,
    env_particles=30,
    env_max_steps=5000,
    agent_step_size=125,      # Must match trained model config
    adaptive_nt_mode=False,   # Must match trained model config
    nt_range=(1, 50),         # Must match trained model config
    num_eval_runs=30,
    checkpoint_base_dir=None, # Expect this from benchmark.py
    # Agent HPs (only needed for init before load)
    hidden_dim=256,
    gamma=1.0,
    tau=0.005,
    alpha=0.2,
    actor_lr=3e-4,
    critic_lr=3e-4
):
    """Main function to load and evaluate the trained SAC agent."""
    # === Use Passed-in Hyperparameters ===

    # --- Use Statically Defined Test Objective Functions ---
    if not test_objective_function_classes:
        print("Error: The test_objective_function_classes list is empty.")
        return

    # --- Determine Model File Path ---
    if checkpoint_base_dir is None:
        script_dir = Path(__file__).parent
        project_root_fallback = script_dir.parents[2]
        checkpoint_base_dir = project_root_fallback / "LLM" / "SAPSO"
        print(f"Warning: checkpoint_base_dir not provided, using default: {checkpoint_base_dir}")

    checkpoint_dir = Path(checkpoint_base_dir) / "checkpoints_sapso_trained_static"
    checkpoint_prefix = "sac_psoenv_adaptive_nt" if adaptive_nt_mode else f"sac_psoenv_fixed_nt{agent_step_size}"
    MODEL_TO_LOAD = checkpoint_dir / f"{checkpoint_prefix}_final.pth"

    print(f"Attempting to load model: {MODEL_TO_LOAD}")
    if not MODEL_TO_LOAD.exists():
        print(f"Error: Model file not found at {MODEL_TO_LOAD}")
        # ... (rest of error message) ...
        return

    # --- Initialize Environment (Placeholder) ---
    print("Creating temporary environment to get dimensions...")
    try:
        temp_obj_func = test_objective_function_classes[0](dim=env_dim, num_particles=env_particles)
        temp_env = PSOEnv(
            obj_func=temp_obj_func,
            max_steps=env_max_steps,
            agent_step_size=agent_step_size,
            adaptive_nt=adaptive_nt_mode,
            nt_range=nt_range,
            convergence_patience=50,
            convergence_threshold_gbest=1e-8,
            convergence_threshold_pbest_std=1e-6
        )
        state_dim = temp_env.observation_space.shape[0]
        action_dim = temp_env.action_space.shape[0]
        temp_env.close()
        print("Temporary environment closed.")
    except Exception as e:
        print(f"Error creating temporary environment: {e}")
        return

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n--- Testing Configuration (from args) ---")
    print(f"Using device: {device}")
    print(f"Loading Model Trained With:")
    print(f"  Adaptive Nt Mode: {adaptive_nt_mode}")
    if not adaptive_nt_mode:
        print(f"  Fixed Agent Step Size (Nt): {agent_step_size}")
    else:
        print(f"  Adaptive Nt Range: {nt_range}")
    # ... (print other relevant args) ...
    print(f"---------------------------\n")

    # --- Initialize Agent ---
    agent = SACAgent(
        state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim,
        gamma=gamma, tau=tau, alpha=alpha, actor_lr=actor_lr, critic_lr=critic_lr,
        device=device, adaptive_nt=adaptive_nt_mode # Use argument
    )

    # --- Load the Trained Agent Model ---
    try:
        agent.load(str(MODEL_TO_LOAD))
        print(f"Successfully loaded trained agent from: {MODEL_TO_LOAD}")
    except Exception as e:
        print(f"Error loading agent model: {e}")
        return

    # === Use passed-in Evaluation Parameters ===
    print(f"\nStarting {num_eval_runs} deterministic evaluation runs per test function...")

    # --- Data Aggregation ---
    evaluation_data = collections.defaultdict(list)
    all_eval_rewards = []
    final_gbests_all_runs = [] # Store final gbest from each run
    eval_start_time = time.time()

    # --- Evaluation Loop ---
    for func_index, func_class in enumerate(test_objective_function_classes):
        print(f"\n--- Evaluating Function {func_index+1}/{len(test_objective_function_classes)}: {func_class.__name__} ---")
        func_rewards = []
        last_gbest_this_func = [] # Store final gbest for runs on this function

        for run in range(num_eval_runs): # Use argument
            print(f"  Starting Evaluation Run {run + 1}/{num_eval_runs} for {func_class.__name__}...")
            current_run_final_gbest = np.inf # Track final gbest for this specific run

            try:
                current_dim = env_dim
                # Add dimension checks if needed

                obj_func_instance = func_class(dim=current_dim, num_particles=env_particles)
                eval_env = PSOEnv(
                    obj_func=obj_func_instance,
                    max_steps=env_max_steps,
                    agent_step_size=agent_step_size,
                    adaptive_nt=adaptive_nt_mode,
                    nt_range=nt_range
                )
                state, _ = eval_env.reset(seed=1000 + func_index * num_eval_runs + run)

            except Exception as e:
                print(f"  Error creating environment for run {run+1}: {e}")
                continue

            run_reward = 0.0
            term, trunc = False, False

            while not term and not trunc:
                if isinstance(state, torch.Tensor):
                    state_np = state.cpu().numpy()
                else:
                    state_np = np.array(state, dtype=np.float32)

                action = agent.select_action(state_np, deterministic=True)
                eval_omega, eval_c1, eval_c2, eval_nt = eval_env._rescale_action(action)
                params_for_turn = [float(eval_omega), float(eval_c1), float(eval_c2)]

                try:
                    next_state, reward, term, trunc, info = eval_env.step(action)
                except Exception as e:
                     print(f"  Error during env.step() in run {run+1}: {e}")
                     print("  Terminating run early.")
                     trunc = True
                     next_state = state
                     reward = 0
                     info = {}

                # Collect metrics
                step_metrics_list = info.get('step_metrics', [])
                run_gbest_in_turn = np.inf # Track gbest within the steps of this turn

                for step_metric_dict in step_metrics_list:
                    pso_step_index = step_metric_dict.get('pso_step', -1)
                    if pso_step_index >= 0 and pso_step_index < eval_env.max_steps:
                        stable_ratio = step_metric_dict.get('stable_ratio', np.nan)
                        feasible_ratio = step_metric_dict.get('feasible_ratio', np.nan)
                        infeasible_ratio = 1.0 - feasible_ratio if not np.isnan(feasible_ratio) else np.nan
                        avg_vel_mag = step_metric_dict.get('avg_velocity_magnitude', step_metric_dict.get('avg_velocity', np.nan))
                        diversity = step_metric_dict.get('diversity', np.nan)
                        gbest_val = step_metric_dict.get('gbest_value', np.nan)

                        # Update the best gbest found *within this agent turn*
                        if gbest_val is not None and not np.isnan(gbest_val):
                             run_gbest_in_turn = min(run_gbest_in_turn, gbest_val)

                        data_point = params_for_turn + [
                            stable_ratio, infeasible_ratio, avg_vel_mag, diversity, gbest_val
                        ]
                        evaluation_data[pso_step_index].append(data_point)

                # Update the final gbest for the run with the best found in this turn
                current_run_final_gbest = min(current_run_final_gbest, run_gbest_in_turn)

                state = next_state
                run_reward += reward

            # End of evaluation run
            func_rewards.append(run_reward)
            if np.isfinite(current_run_final_gbest): # Only store valid final gbests
                 last_gbest_this_func.append(current_run_final_gbest)
            print(f"  Finished Run {run + 1}/{num_eval_runs}. Reward: {run_reward:.4f}, Final GBest: {current_run_final_gbest:.6e}")
            try:
                eval_env.close()
            except Exception as e:
                 print(f"  Warning: Error closing environment for run {run+1}: {e}")

        # Aggregate rewards and final gbests for this function
        if func_rewards:
            all_eval_rewards.extend(func_rewards)
            print(f"--- Function {func_class.__name__} Avg Reward: {np.mean(func_rewards):.4f} +/- {np.std(func_rewards):.4f} ---")
        if last_gbest_this_func:
             final_gbests_all_runs.extend(last_gbest_this_func)
             print(f"--- Function {func_class.__name__} Avg Final GBest: {np.mean(last_gbest_this_func):.6e} +/- {np.std(last_gbest_this_func):.6e} ---")


    # --- End of all evaluation runs ---
    eval_end_time = time.time()
    print(f"\nFinished {num_eval_runs * len(test_objective_function_classes)} total evaluation runs.")
    print(f"Total evaluation time: {eval_end_time - eval_start_time:.2f} seconds.")

    # --- Calculate and Print Overall Final Statistics ---
    print("\n--- Overall Evaluation Results ---")
    if all_eval_rewards:
        mean_reward = np.mean(all_eval_rewards)
        std_reward = np.std(all_eval_rewards)
        print(f"Overall Mean Reward (All Runs & Functions): {mean_reward:.4f} (μ) +/- {std_reward:.4f} (σ)")
    else:
        print("No evaluation reward data collected.")

    if final_gbests_all_runs:
        mean_final_gbest = np.mean(final_gbests_all_runs)
        std_final_gbest = np.std(final_gbests_all_runs)
        print(f"Overall Mean Final GBest (All Runs & Functions): {mean_final_gbest:.6e} (μ) +/- {std_final_gbest:.6e} (σ)")
    else:
        print("No final global best data collected.")
    print("---------------------------------")


    # --- Generate ALL Evaluation Plots ---
    if evaluation_data:
        print("\nGenerating evaluation plots for test functions...")
        plot_output_dir = checkpoint_dir / "test_plots_static"
        os.makedirs(plot_output_dir, exist_ok=True)
        plot_prefix = f"{checkpoint_prefix}_TESTING_STATIC"

        # Call plotting functions
        plot_evaluation_parameters(evaluation_data, env_max_steps, str(plot_output_dir), plot_prefix)
        plot_stable_particles(evaluation_data, env_max_steps, str(plot_output_dir), plot_prefix)
        plot_infeasible_particles(evaluation_data, env_max_steps, str(plot_output_dir), plot_prefix)
        plot_average_velocity(evaluation_data, env_max_steps, str(plot_output_dir), plot_prefix)
        plot_swarm_diversity(evaluation_data, env_max_steps, str(plot_output_dir), plot_prefix)
        plot_gbest_convergence(evaluation_data, env_max_steps, str(plot_output_dir), plot_prefix)
        print(f"Evaluation plots saved in: {plot_output_dir}")
    else:
        print("No evaluation data was collected, skipping plot generation.")


# --- Main execution block removed ---
