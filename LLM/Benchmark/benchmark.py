# benchmark.py - Place in PSO-ToyBox/LLM/Benchmark/
# Defines hyperparameters and calls train/test functions with them.
# Uses the custom logger from LLM.Logs.

import time
import sys  # Import sys for exit
from pathlib import Path
import traceback  # Import traceback for logging errors

# --- Import Logger ---
# Using the specified import path
try:
    from LLM.Logs import logger
    # Import specific helper functions if they exist in logger module
    # Assuming standard names like log_info, log_error etc.
    from LLM.Logs.logger import log_info, log_error, log_warning, log_success, log_header, log
except ImportError:
    # Fallback print if logger fails to import - crucial for basic feedback
    print("ERROR: Logger module not found at 'LLM.Logs.logger'. Please check path and ensure logger.py exists.")
    print("Falling back to standard print statements.")


    # Define dummy functions to avoid crashing the script
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


    def log(msg, mod, col):
        print(f"LOG [{mod}]: {msg}")  # Dummy log function

# --- Import Benchmark Functions ---
# Removed the try-except block as requested
from LLM.Benchmark.test import test_agent
from LLM.Benchmark.train import train_agent

if __name__ == "__main__":
    # Get the module name for logging (using the filename without extension)
    module_name = Path(__file__).stem  # Gets 'benchmark'

    # === Central Hyperparameter Definition ===
    # --- Environment Config ---
    ENV_DIM = 30
    ENV_PARTICLES = 30
    ENV_MAX_STEPS = 5000  # Max PSO steps per env run
    USE_VELOCITY_CLAMPING = True

    # --- Agent/Env Interaction Config ---
    AGENT_STEP_SIZE = 125  # Used for fixed Nt mode
    ADAPTIVE_NT_MODE = False  # Set to True to enable adaptive Nt
    NT_RANGE = (1, 125)  # Range if ADAPTIVE_NT_MODE is True

    # --- Training Config ---
    EPISODES_PER_FUNCTION = 10  # Renamed from NUM_EPISODES for clarity with train.py arg
    BATCH_SIZE = 256
    START_STEPS = 1000  # Total agent steps before learning starts
    UPDATES_PER_STEP = 1  # Agent updates per agent step
    SAVE_FREQ_MULTIPLIER = 4  # Renamed from SAVE_FREQ for clarity with train.py arg

    # --- Testing Config ---
    NUM_EVAL_RUNS = 30  # Number of deterministic runs per test function

    # --- Agent Hyperparameters (can also be passed if needed) ---
    # hidden_dim=256; gamma=1.0; tau=0.005; alpha=0.2; actor_lr=3e-4; critic_lr=3e-4
    # For simplicity, keeping these inside train/test for now, but could be passed

    # --- Checkpoint/Output Config ---
    # Construct paths relative to the project root found earlier
    # (Checkpoint filenames are constructed inside train/test based on config)
    CHECKPOINT_BASE_DIR = "models"  # Relative path for models/checkpoints

    # ========================================
    log_header("Starting SAPSO Benchmark Process", module_name)
    log_header("========================================", module_name)
    log_info("--- Using Hyperparameters ---", module_name)
    log_info(f"  Env Dim: {ENV_DIM}, Particles: {ENV_PARTICLES}, Max Steps: {ENV_MAX_STEPS}", module_name)
    log_info(f"  Adaptive Nt: {ADAPTIVE_NT_MODE}", module_name)
    if not ADAPTIVE_NT_MODE:
        log_info(f"  Fixed Agent Step Size: {AGENT_STEP_SIZE}", module_name)
    else:
        log_info(f"  Adaptive Nt Range: {NT_RANGE}", module_name)
    log_info(f"  Training Episodes per Function: {EPISODES_PER_FUNCTION}", module_name)
    log_info(f"  Batch Size: {BATCH_SIZE}, Start Steps: {START_STEPS}, Updates/Step: {UPDATES_PER_STEP}", module_name)
    log_info(f"  Save Freq Multiplier (Functions): {SAVE_FREQ_MULTIPLIER}", module_name)
    log_info(f"  Evaluation Runs per Test Func: {NUM_EVAL_RUNS}", module_name)
    log_info(f"  Checkpoint Base Dir: {CHECKPOINT_BASE_DIR}", module_name)
    log_info("-----------------------------", module_name)

    train_success = False
    # Step 1: Run the training function, passing hyperparameters
    log_header("Step 1: Training the agent...", module_name)
    start_time_train = time.time()
    try:
        # Call train_agent, assuming it now uses the logger internally
        # If train_agent raises exceptions on failure, this try/except block will catch them.
        # If train_agent uses sys.exit(), the SystemExit exception will be caught.
        train_agent(
            env_dim=ENV_DIM,
            env_particles=ENV_PARTICLES,
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
            use_velocity_clamping=USE_VELOCITY_CLAMPING,
            # Pass other agent hyperparams if needed
        )
        train_success = True  # Assume success if no exception is raised
        log_success("Training function completed.", module_name)
    except SystemExit as e:
        log_warning(f"Training function exited with code {e.code}.", module_name)
        # Decide if exit code 0 should be treated as success
        train_success = (e.code == 0)
    except Exception as e:
        log_error(f"An error occurred during training: {e}", module_name)
        log_error(traceback.format_exc(), module_name)  # Log the full traceback
        train_success = False

    end_time_train = time.time()
    log_info(f"--- Training step finished in {end_time_train - start_time_train:.2f} seconds ---", module_name)

    # Step 2: Run the testing function if training was successful
    if train_success:
        log_header("Step 2: Testing the trained agent...", module_name)
        start_time_test = time.time()
        try:
            # Call test_agent, assuming it now uses the logger internally
            test_agent(
                env_dim=ENV_DIM,
                env_particles=ENV_PARTICLES,
                env_max_steps=ENV_MAX_STEPS,
                agent_step_size=AGENT_STEP_SIZE,  # Must match trained model config
                adaptive_nt_mode=ADAPTIVE_NT_MODE,  # Must match trained model config
                nt_range=NT_RANGE,  # Must match trained model config
                num_eval_runs=NUM_EVAL_RUNS,
                checkpoint_base_dir=CHECKPOINT_BASE_DIR,  # To find the model and save plots
                use_velocity_clamping=USE_VELOCITY_CLAMPING,
                # Pass other agent hyperparams if needed for init before load
            )
            log_success("Testing function completed successfully.", module_name)
        except SystemExit as e:
            log_warning(f"Testing function exited with code {e.code}.", module_name)
        except Exception as e:
            log_error(f"An error occurred during testing: {e}", module_name)
            log_error(traceback.format_exc(), module_name)  # Log the full traceback
            # Log specific failure message for testing
            log_error("Testing function failed.", module_name)

        end_time_test = time.time()
        log_info(f"--- Testing step finished in {end_time_test - start_time_test:.2f} seconds ---", module_name)

    else:
        # Log why testing is skipped
        log_warning("Training did not complete successfully or was skipped. Skipping testing.", module_name)

    log_header("========================================", module_name)
    log_header("SAPSO Benchmark Process Finished", module_name)
    log_header("========================================", module_name)
