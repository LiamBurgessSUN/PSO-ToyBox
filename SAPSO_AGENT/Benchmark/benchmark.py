# benchmark.py - Place in PSO-ToyBox/SAPSO_AGENT/Benchmark/
# Defines hyperparameters and calls train/test functions with them.
# Uses the custom logger from SAPSO_AGENT.Logs.

import time
from pathlib import Path
import traceback  # Import traceback for logging errors
from typing import Dict, Any, Optional, Tuple

# --- Import Logger ---
# Using the specified import path

# Import specific helper functions if they exist in logger module
# Assuming standard names like log_info, log_error etc.
from SAPSO_AGENT.Logs.logger import log_info, log_error, log_warning, log_success, log_header

# --- Import Benchmark Functions ---
# Removed the try-except block as requested
from SAPSO_AGENT.Benchmark.test import test_agent
from SAPSO_AGENT.Benchmark.train import train_agent
from SAPSO_AGENT.CONFIG import *

if __name__ == "__main__":

    # Get the module name for logging (using the filename without extension)
    module_name = Path(__file__).stem  # Gets 'benchmark'

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
    trained_agent = None
    training_results = None
    
    # Step 1: Run the training function, passing hyperparameters
    log_header("Step 1: Training the agent...", module_name)
    start_time_train = time.time()
    try:
        # Call train_agent, assuming it now uses the logger internally
        # If train_agent raises exceptions on failure, this try/except block will catch them.
        # If train_agent uses sys.exit(), the SystemExit exception will be caught.
        trained_agent, training_results = train_agent(
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
        log_success("Training function completed successfully.", module_name)
        
        # Log training summary if results are available
        if training_results:
            log_info("--- Training Results Summary ---", module_name)
            total_episodes = sum(len(results) for results in training_results.values())
            log_info(f"Total episodes completed: {total_episodes}", module_name)
            log_info(f"Functions trained: {list(training_results.keys())}", module_name)
            
            # Calculate overall training statistics
            all_rewards = []
            all_gbests = []
            for func_name, results in training_results.items():
                func_rewards = [r for r, g in results if isinstance(r, (int, float)) and not isinstance(r, bool)]
                func_gbests = [g for r, g in results if isinstance(g, (int, float)) and not isinstance(g, bool)]
                all_rewards.extend(func_rewards)
                all_gbests.extend(func_gbests)
                
                if func_rewards:
                    avg_reward = sum(func_rewards) / len(func_rewards)
                    log_info(f"  {func_name}: {len(func_rewards)} episodes, avg reward: {avg_reward:.4f}", module_name)
            
            if all_rewards:
                overall_avg_reward = sum(all_rewards) / len(all_rewards)
                log_info(f"Overall average training reward: {overall_avg_reward:.4f}", module_name)
            
            if all_gbests:
                best_gbest = min(all_gbests)
                log_info(f"Best GBest achieved during training: {best_gbest:.6e}", module_name)
        
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
            evaluation_data, eval_rewards, eval_gbests = test_agent(
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
            
            # Log testing summary if results are available
            if eval_rewards and eval_gbests:
                log_info("--- Testing Results Summary ---", module_name)
                log_info(f"Total evaluation runs completed: {len(eval_rewards)}", module_name)
                
                if eval_rewards:
                    avg_eval_reward = sum(eval_rewards) / len(eval_rewards)
                    log_info(f"Average evaluation reward: {avg_eval_reward:.4f}", module_name)
                
                if eval_gbests:
                    best_eval_gbest = min(eval_gbests)
                    log_info(f"Best GBest achieved during evaluation: {best_eval_gbest:.6e}", module_name)
                
                # Compare training vs testing performance
                if training_results and eval_rewards:
                    all_train_rewards = []
                    for results in training_results.values():
                        all_train_rewards.extend([r for r, g in results if isinstance(r, (int, float)) and not isinstance(r, bool)])
                    
                    if all_train_rewards:
                        avg_train_reward = sum(all_train_rewards) / len(all_train_rewards)
                        avg_eval_reward = sum(eval_rewards) / len(eval_rewards)
                        log_info(f"Training vs Testing Performance:", module_name)
                        log_info(f"  Training avg reward: {avg_train_reward:.4f}", module_name)
                        log_info(f"  Testing avg reward: {avg_eval_reward:.4f}", module_name)
                        log_info(f"  Performance difference: {avg_eval_reward - avg_train_reward:.4f}", module_name)
            
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

    # Final summary
    log_header("========================================", module_name)
    log_header("SAPSO Benchmark Process Finished", module_name)
    log_header("========================================", module_name)
    
    if train_success:
        log_success("Benchmark completed successfully with both training and testing phases.", module_name)
    else:
        log_warning("Benchmark completed with training failures. Check logs for details.", module_name)
    
    log_info("========================================", module_name)
