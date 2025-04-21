# train.py - Place in PSO-ToyBox/LLM/SAPSO/mains/
# Uses static imports (absolute paths from LLM) for objective functions
# Accepts hyperparameters as arguments
# Trains N episodes for EACH function sequentially
# Calculates and prints final training reward and gbest mean/std

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys
from pathlib import Path

# --- Project Imports (Absolute Paths) ---
try:
    from LLM.RL.ActorCritic.Agent import SACAgent
    from LLM.RL.Replay.ReplayBuffer import ReplayBuffer
    from LLM.SAPSO.Gyms.PSO_GymV2 import PSOEnv
    # Import base class
    from LLM.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction
    # --- Static Imports for Training Objective Functions ---
    from LLM.PSO.ObjectiveFunctions.Training.Ackley import AckleyFunction
    from LLM.PSO.ObjectiveFunctions.Training.Alpine import AlpineFunction
    from LLM.PSO.ObjectiveFunctions.Training.Bohachevsky import BohachevskyFunction
    from LLM.PSO.ObjectiveFunctions.Training.BonyadiMichalewicz import BonyadiMichalewiczFunction
    from LLM.PSO.ObjectiveFunctions.Training.Brown import BrownFunction
    from LLM.PSO.ObjectiveFunctions.Training.CosineMixture import CosineMixtureFunction
    from LLM.PSO.ObjectiveFunctions.Training.DeflectedCorrugatedSpring import DeflectedCorrugatedSpringFunction
    from LLM.PSO.ObjectiveFunctions.Training.Discuss import DiscussFunction
    from LLM.PSO.ObjectiveFunctions.Training.DropWave import DropWaveFunction
    from LLM.PSO.ObjectiveFunctions.Training.EggCrate import EggCrateFunction
    from LLM.PSO.ObjectiveFunctions.Training.EggHolder import EggHolderFunction
    from LLM.PSO.ObjectiveFunctions.Training.Elliptic import EllipticFunction
    from LLM.PSO.ObjectiveFunctions.Training.Exponential import ExponentialFunction
    from LLM.PSO.ObjectiveFunctions.Training.Giunta2D import GiuntaFunction
    from LLM.PSO.ObjectiveFunctions.Training.HolderTable import HolderTable1Function
    from LLM.PSO.ObjectiveFunctions.Training.Levy import Levy3Function
    from LLM.PSO.ObjectiveFunctions.Training.LevyMontalvo import LevyMontalvo2Function
    from LLM.PSO.ObjectiveFunctions.Training.Mishra import Mishra1Function, Mishra4Function
    from LLM.PSO.ObjectiveFunctions.Training.NeedleEye import NeedleEyeFunction
    from LLM.PSO.ObjectiveFunctions.Training.Norweigan import NorwegianFunction # Corrected class name if needed
    from LLM.PSO.ObjectiveFunctions.Training.Pathological import PathologicalFunction
    from LLM.PSO.ObjectiveFunctions.Training.Penalty import Penalty1Function, Penalty2Function
    from LLM.PSO.ObjectiveFunctions.Training.Periodic import PeriodicFunction
    from LLM.PSO.ObjectiveFunctions.Training.Pinter import PinterFunction
    from LLM.PSO.ObjectiveFunctions.Training.Price import Price2Function
    from LLM.PSO.ObjectiveFunctions.Training.Qings import QingsFunction
    from LLM.PSO.ObjectiveFunctions.Training.Quadratic import QuadricFunction
    from LLM.PSO.ObjectiveFunctions.Training.Quintic import QuinticFunction
    from LLM.PSO.ObjectiveFunctions.Training.Rana import RanaFunction
    from LLM.PSO.ObjectiveFunctions.Training.Rastrgin import RastriginFunction
    from LLM.PSO.ObjectiveFunctions.Training.Ripple import Ripple25Function
    from LLM.PSO.ObjectiveFunctions.Training.Rosenbrock import RosenbrockFunction
    from LLM.PSO.ObjectiveFunctions.Training.Salomon import SalomonFunction
    from LLM.PSO.ObjectiveFunctions.Training.Schubert import Schubert4Function
    from LLM.PSO.ObjectiveFunctions.Training.Schwefel import Schwefel1Function
    from LLM.PSO.ObjectiveFunctions.Training.Sinusoidal import SinusoidalFunction
    from LLM.PSO.ObjectiveFunctions.Training.Step import StepFunction3
    from LLM.PSO.ObjectiveFunctions.Training.Trid import TridFunction
    from LLM.PSO.ObjectiveFunctions.Training.Trigonometric import TrigonometricFunction
    from LLM.PSO.ObjectiveFunctions.Training.Vincent import VincentFunction
    from LLM.PSO.ObjectiveFunctions.Training.Weierstrass import WeierstrassFunction
    from LLM.PSO.ObjectiveFunctions.Training.XinSheYang import XinSheYang1Function, XinSheYang2Function

except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure the script is run from a directory where the 'LLM' package is discoverable,")
    print("or that the project root directory (containing LLM) is in your PYTHONPATH.")
    sys.exit(1)
# --- End Project Imports ---


# --- Manually Create List of Objective Function Classes ---
objective_function_classes = [
    AckleyFunction, AlpineFunction, BohachevskyFunction, BonyadiMichalewiczFunction,
    BrownFunction, CosineMixtureFunction, DeflectedCorrugatedSpringFunction,
    DiscussFunction, DropWaveFunction, EggCrateFunction, EggHolderFunction,
    EllipticFunction, ExponentialFunction, GiuntaFunction, HolderTable1Function,
    Levy3Function, LevyMontalvo2Function, Mishra1Function, Mishra4Function,
    NeedleEyeFunction, NorwegianFunction, PathologicalFunction, Penalty1Function,
    Penalty2Function, PeriodicFunction, PinterFunction, Price2Function,
    QingsFunction, QuadricFunction, QuinticFunction, RanaFunction, RastriginFunction,
    Ripple25Function, RosenbrockFunction, SalomonFunction, Schubert4Function,
    Schwefel1Function, SinusoidalFunction, StepFunction3, TridFunction,
    TrigonometricFunction, VincentFunction, WeierstrassFunction, XinSheYang1Function,
    XinSheYang2Function
]
# --- End Manual List ---


# --- Main Training Function (Accepts Arguments) ---
def train_agent(
    env_dim=30,
    env_particles=30,
    env_max_steps=5000,
    agent_step_size=125,
    adaptive_nt_mode=False,
    nt_range=(1, 50),
    episodes_per_function=5, # Changed from num_episodes
    batch_size=256,
    start_steps=1000,
    updates_per_step=1,
    save_freq_multiplier=4, # Save every N functions trained
    checkpoint_base_dir=None, # Expect this from benchmark.py
    # Agent HPs (can be arguments too)
    hidden_dim=256,
    gamma=1.0,
    tau=0.005,
    alpha=0.2,
    actor_lr=3e-4,
    critic_lr=3e-4
):
    """Main function to train the SAC agent, running N episodes for each objective function."""
    # === Use Passed-in Hyperparameters ===

    # --- Use Statically Defined Objective Functions ---
    if not objective_function_classes:
        print("Error: The objective_function_classes list is empty.")
        return

    # --- Initialize Environment (using a placeholder initially) ---
    print("Creating temporary environment to get dimensions...")
    try:
        temp_obj_func = objective_function_classes[0](dim=env_dim, num_particles=env_particles)
        temp_env = PSOEnv(
            obj_func=temp_obj_func,
            max_steps=env_max_steps,
            agent_step_size=agent_step_size,
            adaptive_nt=adaptive_nt_mode,
            nt_range=nt_range
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
    print(f"\n--- Training Configuration (from args) ---")
    print(f"Using device: {device}")
    print(f"Adaptive Nt Mode: {adaptive_nt_mode}")
    if not adaptive_nt_mode:
        print(f"Fixed Agent Step Size (Nt): {agent_step_size}")
    else:
        print(f"Adaptive Nt Range: {nt_range}")
    print(f"Episodes per Function: {episodes_per_function}")
    # ... (print other passed-in args) ...
    print(f"----------------------------\n")

    # --- Initialize Agent ---
    agent = SACAgent(
        state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim,
        gamma=gamma, tau=tau, alpha=alpha, actor_lr=actor_lr, critic_lr=critic_lr,
        device=device, adaptive_nt=adaptive_nt_mode
    )

    # --- Initialize Replay Buffer ---
    buffer_capacity = 1_000_000 # Could be an argument
    buffer = ReplayBuffer(
        capacity=buffer_capacity, state_dim=state_dim, action_dim=action_dim, device=device
    )

    # === Use passed-in Training Hyperparameters ===
    # episodes_per_function, batch_size, start_steps, updates_per_step, save_freq_multiplier used below

    # --- Tracking Variables ---
    # Store tuples of (reward, final_gbest) for each episode
    results_log = {} # {func_name: [(ep1_reward, ep1_gbest), (ep2_reward, ep2_gbest), ...]}
    global_step_count = 0
    total_agent_steps = 0
    total_episodes_run = 0

    # --- Checkpoint Setup ---
    if checkpoint_base_dir is None:
        script_dir = Path(__file__).parent
        project_root_fallback = script_dir.parents[2]
        checkpoint_base_dir = project_root_fallback / "LLM" / "SAPSO"
        print(f"Warning: checkpoint_base_dir not provided, using default: {checkpoint_base_dir}")

    checkpoint_dir = Path(checkpoint_base_dir) / "checkpoints_sapso_trained_static"
    checkpoint_prefix = "sac_psoenv_adaptive_nt" if adaptive_nt_mode else f"sac_psoenv_fixed_nt{agent_step_size}"
    checkpoint_file = checkpoint_dir / f"{checkpoint_prefix}_checkpoint.pth"
    final_model_file = checkpoint_dir / f"{checkpoint_prefix}_final.pth"
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved in: {checkpoint_dir}")

    # --- Main Training Loop (Nested) ---
    print("Starting training...")
    train_start_time = time.time()

    # Outer loop: Iterate through each objective function
    for func_index, selected_func_class in enumerate(objective_function_classes):
        func_name = selected_func_class.__name__
        print(f"\n===== Training on Function {func_index + 1}/{len(objective_function_classes)}: {func_name} =====")
        results_log[func_name] = [] # Initialize list for this function

        # Inner loop: Run N episodes for the current function
        for episode_num in range(episodes_per_function): # Use argument
            total_episodes_run += 1
            print(f"\n--- Episode {episode_num + 1}/{episodes_per_function} (Total Ep: {total_episodes_run}) | Function: {func_name} ---")

            train_env = None # Ensure env is defined in scope
            try:
                current_dim = env_dim
                if func_name == "GiuntaFunction":
                     current_dim = 2
                     print(f"  Adjusting dimension to {current_dim} for {func_name}")

                obj_func_instance = selected_func_class(dim=current_dim, num_particles=env_particles)

                train_env = PSOEnv(
                    obj_func=obj_func_instance,
                    max_steps=env_max_steps,
                    agent_step_size=agent_step_size,
                    adaptive_nt=adaptive_nt_mode,
                    nt_range=nt_range,
                    convergence_patience=50,
                    convergence_threshold_gbest=1e-8,
                    convergence_threshold_pbest_std=1e-6
                )
                state, _ = train_env.reset(seed=total_episodes_run)

            except Exception as e:
                print(f"Error creating environment for {func_name}, episode {episode_num+1}: {e}")
                print("Skipping this episode.")
                continue

            episode_reward = 0.0
            terminated, truncated = False, False
            episode_agent_steps = 0
            episode_final_gbest = np.inf # Track best gbest found in this episode

            # --- Innermost Loop (Agent Steps within Episode) ---
            while not terminated and not truncated:
                if total_agent_steps < start_steps:
                    action = train_env.action_space.sample()
                else:
                    if isinstance(state, torch.Tensor):
                        state_np = state.cpu().numpy()
                    else:
                        state_np = np.array(state, dtype=np.float32)
                    action = agent.select_action(state_np)

                try:
                    next_state, reward, terminated, truncated, info = train_env.step(action)
                    # Update episode's best gbest if available in info
                    step_metrics = info.get('step_metrics', [])
                    for metric_dict in step_metrics:
                         gbest_val = metric_dict.get('gbest_value', np.inf)
                         if gbest_val is not None and not np.isnan(gbest_val):
                              episode_final_gbest = min(episode_final_gbest, gbest_val)

                except Exception as e:
                     print(f"Error during env.step() in episode {episode_num+1}: {e}")
                     print("Terminating episode early.")
                     truncated = True
                     next_state = state
                     reward = 0
                     info = {}

                # Update counters
                episode_agent_steps += 1
                total_agent_steps += 1
                pso_steps_this_turn = info.get('steps_taken', 0)
                current_nt_used = info.get('nt', agent_step_size if not adaptive_nt_mode else nt_range[0])
                global_step_count += pso_steps_this_turn

                # Store experience
                done_flag = terminated or truncated
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
                print_freq_train = 200
                # if episode_agent_steps % print_freq_train == 0:
                #      print(f"    Ep {episode_num+1} | Agt Step {episode_agent_steps} "
                #            f"| PSO Step {train_env.current_step}/{train_env.max_steps} "
                #            f"| Buf {len(buffer)} | R: {reward:.4f} (nt={current_nt_used})")

            # --- End of Episode ---
            # Get final gbest from the environment state if not found during steps
            if train_env and hasattr(train_env, 'pso') and np.isinf(episode_final_gbest):
                 episode_final_gbest = min(episode_final_gbest, train_env.pso.gbest_value)

            results_log[func_name].append((episode_reward, episode_final_gbest)) # Store tuple

            print(f"--- Episode {episode_num + 1} Finished ---")
            print(f"  Agent Steps: {episode_agent_steps}")
            print(f"  Total Reward: {episode_reward:.4f}")
            print(f"  Final GBest: {episode_final_gbest:.6e}")
            print(f"  Global PSO Steps: {global_step_count}")
            print(f"  Total Agent Steps: {total_agent_steps}")

            try:
                if train_env: train_env.close()
            except Exception as e:
                print(f"Warning: Error closing environment for episode {episode_num+1}: {e}")

        # --- End of Episodes for Current Function ---
        print(f"===== Function {func_name} Training Complete =====")
        if results_log[func_name]:
             func_rewards = [r for r, g in results_log[func_name]]
             func_gbests = [g for r, g in results_log[func_name] if np.isfinite(g)]
             print(f"  Avg Reward over {episodes_per_function} episodes: {np.mean(func_rewards):.4f}")
             if func_gbests:
                  print(f"  Avg Final GBest over {len(func_gbests)} episodes: {np.mean(func_gbests):.6e}")
             else:
                  print(f"  No finite Final GBest values recorded for this function.")


        # --- Periodic Checkpoint Saving (Based on Functions Completed) ---
        if (func_index + 1) % save_freq_multiplier == 0: # Use argument
            try:
                agent.save(str(checkpoint_file))
                print(f"Checkpoint saved after function {func_index + 1} ({func_name})")
            except Exception as e:
                print(f"Warning: Could not save checkpoint during training: {e}")

    # --- End of Training Loop (All Functions) ---
    train_end_time = time.time()
    print(f"\nTraining finished. Total time: {train_end_time - train_start_time:.2f} seconds.")
    print(f"Total Episodes Run: {total_episodes_run}")

    # --- Calculate and Print Overall Final Statistics ---
    all_final_rewards = []
    all_final_gbests = []
    for func_name, results in results_log.items():
        all_final_rewards.extend([r for r, g in results])
        all_final_gbests.extend([g for r, g in results if np.isfinite(g)])

    print("\n--- Overall Training Results ---")
    if all_final_rewards:
        mean_reward = np.mean(all_final_rewards)
        std_reward = np.std(all_final_rewards)
        # Variance = std**2
        var_reward = np.var(all_final_rewards)
        print(f"Overall Mean Training Reward (All Eps & Funcs): {mean_reward:.4f} (μ) +/- {std_reward:.4f} (σ)")
        print(f"Overall Training Reward Variance: {var_reward:.6f}")

    else:
        print("No training reward data collected.")

    if all_final_gbests:
        mean_final_gbest = np.mean(all_final_gbests)
        std_final_gbest = np.std(all_final_gbests)
        # Variance = std**2
        var_final_gbest = np.var(all_final_gbests)
        print(f"Overall Mean Final GBest (All Valid Eps & Funcs): {mean_final_gbest:.6e} (μ) +/- {std_final_gbest:.6e} (σ)")
        print(f"Overall Final GBest Variance: {var_final_gbest:.6e}")
    else:
        print("No final global best data collected.")
    print("---------------------------------")


    # --- Save Final Model ---
    try:
        agent.save(str(final_model_file))
        print(f"Final trained model saved to {final_model_file}")
    except Exception as e:
        print(f"Warning: Could not save final checkpoint: {e}")

    # --- Plot Training Reward Curve (Average across functions per 'epoch') ---
    # Plotting average reward per function
    avg_rewards_per_func = {}
    for func_name, results in results_log.items():
        if results:
            avg_rewards_per_func[func_name] = np.mean([r for r,g in results])

    if avg_rewards_per_func:
        plt.figure(figsize=(12, 6))
        function_names = list(avg_rewards_per_func.keys())
        avg_rewards = list(avg_rewards_per_func.values())
        plt.bar(function_names, avg_rewards)
        mode_str = '(Adaptive Nt)' if adaptive_nt_mode else f'(Fixed Nt={agent_step_size})'
        plt.title(f"SAC Avg Reward per Function ({episodes_per_function} eps each) {mode_str}")
        plt.xlabel("Objective Function")
        plt.ylabel(f"Average Reward over {episodes_per_function} Episodes")
        plt.xticks(rotation=90)
        plt.grid(axis='y')
        plt.tight_layout()
        plot_filename = checkpoint_dir / f"{checkpoint_prefix}_train_rewards_per_func_static.png"
        try:
            plt.savefig(str(plot_filename))
            print(f"Training reward plot (per function) saved to {plot_filename}")
        except Exception as e:
            print(f"Warning: Could not save training reward plot: {e}")
        plt.close() # Close the figure to free memory

# --- Main execution block removed ---
