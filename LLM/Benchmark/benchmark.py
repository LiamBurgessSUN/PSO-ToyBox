# benchmark.py - Place in PSO-ToyBox/LLM/SAPSO/mains/
# Defines hyperparameters and calls train/test functions with them.

import time
from pathlib import Path

from LLM.Benchmark.test import test_agent
from LLM.Benchmark.train import train_agent

if __name__ == "__main__":
    # === Central Hyperparameter Definition ===
    # --- Environment Config ---
    ENV_DIM = 30
    ENV_PARTICLES = 30
    ENV_MAX_STEPS = 5000 # Max PSO steps per env run

    # --- Agent/Env Interaction Config ---
    AGENT_STEP_SIZE = 125 # Used for fixed Nt mode
    ADAPTIVE_NT_MODE = False # Set to True to enable adaptive Nt
    NT_RANGE = (1, 50)    # Range if ADAPTIVE_NT_MODE is True

    # --- Training Config ---
    NUM_EPISODES = 10    # Total number of training episodes
    BATCH_SIZE = 256
    START_STEPS = 1000    # Total agent steps before learning starts
    UPDATES_PER_STEP = 1  # Agent updates per agent step
    SAVE_FREQ = 4       # Save checkpoint every N episodes

    # --- Testing Config ---
    NUM_EVAL_RUNS = 30    # Number of deterministic runs per test function

    # --- Agent Hyperparameters (can also be passed if needed) ---
    # hidden_dim=256; gamma=1.0; tau=0.005; alpha=0.2; actor_lr=3e-4; critic_lr=3e-4
    # For simplicity, keeping these inside train/test for now, but could be passed

    # --- Checkpoint/Output Config ---
    # Construct paths relative to the project root found earlier
    # (Checkpoint filenames are constructed inside train/test based on config)

    CHECKPOINT_BASE_DIR = "models"

    # ========================================
    print("Starting SAPSO Benchmark Process (Direct Import with Params)")
    print("========================================")
    print("--- Using Hyperparameters ---")
    print(f"  Env Dim: {ENV_DIM}, Particles: {ENV_PARTICLES}, Max Steps: {ENV_MAX_STEPS}")
    print(f"  Adaptive Nt: {ADAPTIVE_NT_MODE}")
    if not ADAPTIVE_NT_MODE:
        print(f"  Fixed Agent Step Size: {AGENT_STEP_SIZE}")
    else:
        print(f"  Adaptive Nt Range: {NT_RANGE}")
    print(f"  Training Episodes: {NUM_EPISODES}, Batch Size: {BATCH_SIZE}")
    print(f"  Evaluation Runs per Test Func: {NUM_EVAL_RUNS}")
    print(f"  Checkpoint Base Dir: {CHECKPOINT_BASE_DIR}")
    print("-----------------------------")


    train_success = False
    # Step 1: Run the training function, passing hyperparameters
    print("\nStep 1: Training the agent...")
    start_time_train = time.time()
    try:
        train_agent(
            env_dim=ENV_DIM,
            env_particles=ENV_PARTICLES,
            env_max_steps=ENV_MAX_STEPS,
            agent_step_size=AGENT_STEP_SIZE,
            adaptive_nt_mode=ADAPTIVE_NT_MODE,
            nt_range=NT_RANGE,
            episodes_per_function=NUM_EPISODES,
            batch_size=BATCH_SIZE,
            start_steps=START_STEPS,
            updates_per_step=UPDATES_PER_STEP,
            save_freq_multiplier=SAVE_FREQ,
            checkpoint_base_dir=CHECKPOINT_BASE_DIR
            # Pass other agent hyperparams if needed
        )
        train_success = True
        print("\nTraining function completed.")
    except SystemExit as e:
        print(f"\nTraining function exited with code {e.code}.")
        train_success = (e.code == 0)
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        train_success = False

    end_time_train = time.time()
    print(f"--- Training step finished in {end_time_train - start_time_train:.2f} seconds ---")


    # Step 2: Run the testing function if training was successful
    if train_success:
        print("\nStep 2: Testing the trained agent...")
        start_time_test = time.time()
        try:
            test_agent(
                env_dim=ENV_DIM,
                env_particles=ENV_PARTICLES,
                env_max_steps=ENV_MAX_STEPS,
                agent_step_size=AGENT_STEP_SIZE, # Must match trained model config
                adaptive_nt_mode=ADAPTIVE_NT_MODE, # Must match trained model config
                nt_range=NT_RANGE, # Must match trained model config
                num_eval_runs=NUM_EVAL_RUNS,
                checkpoint_base_dir=CHECKPOINT_BASE_DIR # To find the model and save plots
                # Pass other agent hyperparams if needed for init before load
            )
            print("\nTesting function completed successfully.")
        except SystemExit as e:
             print(f"\nTesting function exited with code {e.code}.")
        except Exception as e:
            print(f"\nAn error occurred during testing: {e}")
            import traceback
            traceback.print_exc()
            print("\nTesting function failed.")

        end_time_test = time.time()
        print(f"--- Testing step finished in {end_time_test - start_time_test:.2f} seconds ---")

    else:
        print("\nTraining failed or exited abnormally. Skipping testing.")

    print("\n========================================")
    print("SAPSO Benchmark Process Finished")
    print("========================================")
