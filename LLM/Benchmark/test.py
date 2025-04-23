import torch
import numpy as np
import time
import collections
import traceback  # For logging exceptions
from pathlib import Path
from LLM.SAPSO.RL.ActorCritic.Agent import SACAgent
from LLM.SAPSO.Environment.Environment import Environment
from LLM.SAPSO.Graphics.graphing import (
    plot_evaluation_parameters,
    plot_stable_particles,
    plot_infeasible_particles,
    plot_average_velocity,
    plot_swarm_diversity,
    plot_gbest_convergence
)

from LLM.Logs.logger import *
from LLM.SAPSO.PSO.ObjectiveFunctions.Testing.Loader import test_objective_function_classes

# --- Main Testing Function (Accepts Arguments) ---
def test_agent(
        env_dim=30,
        env_particles=30,
        env_max_steps=5000,
        agent_step_size=125,
        adaptive_nt_mode=False,
        nt_range=(1, 50),
        num_eval_runs=30,
        checkpoint_base_dir=None,
        hidden_dim=256,
        gamma=1.0,
        tau=0.005,
        alpha=0.2,
        actor_lr=3e-4,
        critic_lr=3e-4,
        v_clamp_ratio=0.2,
        use_velocity_clamping=True,
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

    # --- Determine Model File Path ---
    if checkpoint_base_dir is None:
        script_dir = Path(__file__).parent
        project_root_fallback = script_dir.parents[1]
        checkpoint_base_dir = project_root_fallback / "SAPSO" / "checkpoints"
        log_warning(f"checkpoint_base_dir not provided, using default: {checkpoint_base_dir}", module_name)

    mode_suffix = "adaptive_nt" if adaptive_nt_mode else f"fixed_nt{agent_step_size}"
    checkpoint_dir = Path(checkpoint_base_dir) / f"checkpoints_sapso_vectorized_{mode_suffix}"
    checkpoint_prefix = f"sac_psoenv_vectorized_{mode_suffix}"
    MODEL_TO_LOAD = checkpoint_dir / f"{checkpoint_prefix}_final.pth"

    log_info(f"Attempting to load model: {MODEL_TO_LOAD}", module_name)
    if not MODEL_TO_LOAD.exists():
        log_error(f"Model file not found at {MODEL_TO_LOAD}", module_name)
        log_error("Ensure training ran with the same config and checkpoint base dir is correct.", module_name)
        return

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
        device=device, adaptive_nt=adaptive_nt_mode
    )

    # --- Load the Trained Agent Model ---
    try:
        agent.load(str(MODEL_TO_LOAD))
        log_success(f"Successfully loaded trained agent from: {MODEL_TO_LOAD}", module_name)
    except Exception as e:
        log_error(f"Error loading agent model: {e}", module_name)
        log_error(traceback.format_exc(), module_name)
        return

    log_header(f"Starting {num_eval_runs} deterministic evaluation runs per test function...", module_name)

    # --- Data Aggregation ---
    evaluation_data = collections.defaultdict(list)
    all_eval_rewards = []
    final_gbests_all_runs = []
    eval_start_time = time.time()

    # --- Evaluation Loop ---
    for func_index, func_class in enumerate(test_objective_function_classes):
        func_name = func_class.__name__
        log_info(f"--- Evaluating Function {func_index + 1}/{len(test_objective_function_classes)}: {func_name} ---", module_name)
        func_rewards = []
        last_gbest_this_func = []

        for run in range(num_eval_runs):
            log_debug(f"  Starting Evaluation Run {run + 1}/{num_eval_runs} for {func_name}...", module_name)
            run_best_gbest = np.inf

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
                )
                state, _ = eval_env.reset(seed=1000 + func_index * num_eval_runs + run)
                run_best_gbest = eval_env.pso.gbest_value
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
                    next_state, reward, term, trunc, info = eval_env.step(action)
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
        plot_output_dir = checkpoint_dir / "test_plots_vectorized"
        os.makedirs(plot_output_dir, exist_ok=True)
        plot_prefix = f"{checkpoint_prefix}_TESTING_VECTORIZED"

        try:
            plot_evaluation_parameters(evaluation_data, env_max_steps, str(plot_output_dir), plot_prefix)
            plot_stable_particles(evaluation_data, env_max_steps, str(plot_output_dir), plot_prefix) # Index 3
            plot_infeasible_particles(evaluation_data, env_max_steps, str(plot_output_dir), plot_prefix)
            plot_average_velocity(evaluation_data, env_max_steps, str(plot_output_dir), plot_prefix)
            plot_swarm_diversity(evaluation_data, env_max_steps, str(plot_output_dir), plot_prefix)
            plot_gbest_convergence(evaluation_data, env_max_steps, str(plot_output_dir), plot_prefix)
            log_success(f"Evaluation plots saved in: {plot_output_dir}", module_name)
        except Exception as e:
            log_error(f"Error generating plots: {e}", module_name)
            log_error(traceback.format_exc(), module_name)
            log_warning("Plot generation failed. Ensure graphing.py is compatible.", module_name)
    else:
        log_warning("No evaluation data was collected, skipping plot generation.", module_name)

# --- Main execution block removed ---
