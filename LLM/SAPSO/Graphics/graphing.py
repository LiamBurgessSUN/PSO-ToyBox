# File: LLM/SAPSO/Graphics/graphing.py
# Refactored to use the logger module
# Added debug logs to _aggregate_metric and _plot_metric_mean_std
# Adjusted y-limits for stable particles plot

import matplotlib.pyplot as plt
import numpy as np
import os
import traceback # For logging exceptions
from pathlib import Path # To get module name

# --- Import Logger ---
try:
    from ...Logs import logger # Adjust relative path if needed
    from ...Logs.logger import log_info, log_error, log_warning, log_success, log_header, log_debug
except ImportError:
    print("ERROR: Logger module not found at 'LLM.Logs.logger'. Please check path.")
    print("Falling back to standard print statements.")
    def log_info(msg, mod): print(f"INFO [{mod}]: {msg}")
    def log_error(msg, mod): print(f"ERROR [{mod}]: {msg}")
    def log_warning(msg, mod): print(f"WARNING [{mod}]: {msg}")
    def log_success(msg, mod): print(f"SUCCESS [{mod}]: {msg}")
    def log_header(msg, mod): print(f"HEADER [{mod}]: {msg}")
    def log_debug(msg, mod): print(f"DEBUG [{mod}]: {msg}") # Optional debug

# --- Module Name for Logging ---
module_name = Path(__file__).stem # Gets 'graphing'


# === Helper function to aggregate data ===
def _aggregate_metric(eval_data, max_steps, metric_index, metric_name):
    """Helper to extract and aggregate a specific metric from eval_data."""
    log_debug(f"Aggregating '{metric_name}' data (Index: {metric_index})...", module_name)
    steps = sorted(eval_data.keys())
    if not steps:
        log_warning(f"No evaluation data steps found for metric '{metric_name}'.", module_name)
        return None, None, None, 0

    metric_values = [[] for _ in range(max_steps)]
    num_runs = 0
    try:
        first_step_with_data = next((step for step in steps if eval_data.get(step)), -1)
        if first_step_with_data != -1 and eval_data[first_step_with_data]:
             num_runs = len(eval_data[first_step_with_data])
        else:
             log_warning(f"No steps with actual data found for metric '{metric_name}'. Cannot determine number of runs.", module_name)
             num_runs = 0
    except Exception as e:
        log_warning(f"Could not determine number of runs for metric '{metric_name}': {e}", module_name)
        num_runs = 0

    valid_data_points_count = 0
    for step in steps:
        if step < max_steps and step in eval_data:
            step_data = eval_data[step]
            for run_data_point in step_data:
                if run_data_point is not None and len(run_data_point) > metric_index:
                    metric_val = run_data_point[metric_index]
                    # Check if it's a valid number (not None, NaN, or Inf)
                    if isinstance(metric_val, (int, float)) and np.isfinite(metric_val):
                        metric_values[step].append(metric_val)
                        valid_data_points_count += 1
                    # else: # Optional: Log invalid values found
                    #    log_debug(f"Invalid/Non-finite value '{metric_val}' (type: {type(metric_val)}) found at step {step} for metric {metric_name}", module_name)

    log_debug(f"Found {valid_data_points_count} valid data points for '{metric_name}'.", module_name)

    means, stds = [], []
    valid_steps = []

    for i in range(max_steps):
        step_vals = metric_values[i]
        if len(step_vals) > 0: # Changed condition to > 0 to include single points
            valid_steps.append(i)
            means.append(np.mean(step_vals))
            # Calculate std dev only if more than 1 point, otherwise it's 0
            stds.append(np.std(step_vals) if len(step_vals) > 1 else 0.0)

    if not valid_steps:
        log_warning(f"Not enough valid data points across steps/runs for metric '{metric_name}'. Cannot plot.", module_name)
        return None, None, None, num_runs

    # --- DEBUG LOG ---
    log_debug(f"Aggregation complete for '{metric_name}'. Found {len(valid_steps)} steps with data.", module_name)
    log_debug(f"  First 5 means for '{metric_name}': {means[:5]}", module_name)
    log_debug(f"  First 5 stds for '{metric_name}': {stds[:5]}", module_name)
    # --- END DEBUG LOG ---

    return np.array(valid_steps), np.array(means), np.array(stds), num_runs

# === Generic plotting function ===
def _plot_metric_mean_std(valid_steps, means, stds, num_runs,
                         title_base, ylabel, color,
                         checkpoint_dir, filename_suffix,
                         max_steps, use_log_scale=False,
                         y_limit_bottom=None, y_limit_top=None):
    """Generic function to plot mean and std dev for a metric."""
    if valid_steps is None or len(valid_steps) == 0:
        return

    log_info(f"Generating plot: {title_base}", module_name)
    # --- DEBUG LOG ---
    log_debug(f"Plotting '{title_base}': {len(valid_steps)} valid steps.", module_name)
    log_debug(f"  Means range: [{np.min(means):.2f}, {np.max(means):.2f}]", module_name)
    log_debug(f"  Stds range: [{np.min(stds):.2f}, {np.max(stds):.2f}]", module_name)
    # --- END DEBUG LOG ---

    plt.figure(figsize=(10, 6))
    title = f'{title_base} ({num_runs} Runs)'
    plot_label = f'{ylabel}'

    # Plot mean line
    plt.plot(valid_steps, means, label=plot_label, color=color, linewidth=1.5)
    # Plot shaded standard deviation area - check if stds are non-zero before plotting fill
    if np.any(stds > 1e-9): # Check if there's any significant std dev
        plt.fill_between(valid_steps, means - stds, means + stds,
                         color=color, alpha=0.2)
    else:
        log_debug(f"  Skipping fill_between for '{title_base}' as std dev is near zero.", module_name)


    plt.xlabel("PSO Time Steps")
    plt.ylabel(ylabel + (" (log scale)" if use_log_scale else ""))
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True, which='both' if use_log_scale else 'major', linestyle='--')

    current_bottom, current_top = plt.ylim()

    if use_log_scale:
        plt.yscale('log')
        safe_means = means[means > 0]
        min_positive_mean = np.min(safe_means) if len(safe_means) > 0 else None

        if y_limit_bottom is None:
            if min_positive_mean is not None and min_positive_mean > 0:
                 y_limit_bottom = min_positive_mean * 0.1
            else:
                 y_limit_bottom = 1e-6
        y_limit_bottom = max(y_limit_bottom, np.finfo(float).tiny)

    # Set final limits based on input or defaults
    final_bottom = y_limit_bottom if y_limit_bottom is not None else current_bottom
    final_top = y_limit_top if y_limit_top is not None else current_top

    # Adjust limits slightly if they are identical (constant line)
    # Check if the *data* range is near zero, not just the limits
    data_range = np.max(means) - np.min(means) if len(means) > 0 else 0
    if data_range < 1e-9: # Check if data is effectively constant
        log_debug(f"Adjusting y-limits for constant data in '{title_base}'.", module_name)
        # Use the actual mean value for padding calculation
        mean_val = means[0] if len(means) > 0 else 0
        padding = max(abs(mean_val * 0.1), 0.1) # Add 10% padding or 0.1
        final_bottom = mean_val - padding
        final_top = mean_val + padding
        # Ensure limits remain sensible if mean is near 0 or 1 for ratio plots
        if y_limit_bottom is not None: final_bottom = max(final_bottom, y_limit_bottom)
        if y_limit_top is not None: final_top = min(final_top, y_limit_top)


    # Final check to prevent invalid limits
    if final_bottom >= final_top:
        log_warning(f"Invalid y-limits after padding for plot '{title}': bottom ({final_bottom}) >= top ({final_top}). Using default range.", module_name)
        # Fallback to default limits or a small range around the mean
        mean_val = means[0] if len(means) > 0 else 0
        final_bottom = mean_val - 0.1
        final_top = mean_val + 0.1

    plt.ylim(bottom=final_bottom, top=final_top)
    plt.xlim(0, max_steps)
    plt.tight_layout()

    os.makedirs(checkpoint_dir, exist_ok=True)
    plot_filename = os.path.join(checkpoint_dir, filename_suffix)
    try:
        plt.savefig(plot_filename)
        log_success(f"Plot saved to {plot_filename}", module_name)
    except Exception as e:
        log_error(f"Could not save plot {plot_filename}: {e}", module_name)
        log_error(traceback.format_exc(), module_name)
    plt.close(plt.gcf())


# === Specific Plotting Functions ===

def plot_evaluation_parameters(eval_data, max_steps, checkpoint_dir, checkpoint_prefix):
    """Plots mean/std dev for Control Parameters (omega, c1, c2)."""
    log_info("Extracting and plotting Control Parameter data...", module_name)
    steps = sorted(eval_data.keys())
    if not steps:
        log_warning("No evaluation data found for Control Parameter plotting.", module_name)
        return

    omegas = [[] for _ in range(max_steps)]
    c1s    = [[] for _ in range(max_steps)]
    c2s    = [[] for _ in range(max_steps)]
    num_runs = 0
    try:
        first_step_with_data = next((step for step in steps if eval_data.get(step)), -1)
        if first_step_with_data != -1 and eval_data[first_step_with_data]:
             num_runs = len(eval_data[first_step_with_data])
    except Exception as e:
        log_warning(f"Could not determine number of runs for CPs: {e}", module_name)

    for step in steps:
        if step < max_steps and step in eval_data:
            for run_data in eval_data[step]:
                if run_data is not None and len(run_data) >= 3:
                    if run_data[0] is not None and np.isfinite(run_data[0]): omegas[step].append(run_data[0])
                    if run_data[1] is not None and np.isfinite(run_data[1]): c1s[step].append(run_data[1])
                    if run_data[2] is not None and np.isfinite(run_data[2]): c2s[step].append(run_data[2])

    mean_w, std_w, mean_c1, std_c1, mean_c2, std_c2 = [],[],[],[],[],[]
    valid_steps = []
    for i in range(max_steps):
        valid_w = len(omegas[i]) > 0
        valid_c1 = len(c1s[i]) > 0
        valid_c2 = len(c2s[i]) > 0

        if valid_w or valid_c1 or valid_c2:
             valid_steps.append(i)
             mean_w.append(np.mean(omegas[i]) if valid_w else np.nan)
             std_w.append(np.std(omegas[i]) if len(omegas[i]) > 1 else 0.0 if valid_w else np.nan)
             mean_c1.append(np.mean(c1s[i]) if valid_c1 else np.nan)
             std_c1.append(np.std(c1s[i]) if len(c1s[i]) > 1 else 0.0 if valid_c1 else np.nan)
             mean_c2.append(np.mean(c2s[i]) if valid_c2 else np.nan)
             std_c2.append(np.std(c2s[i]) if len(c2s[i]) > 1 else 0.0 if valid_c2 else np.nan)

    if not valid_steps:
        log_warning("Not enough valid data found for Control Parameter plot.", module_name)
        return

    valid_steps = np.array(valid_steps)
    mean_w, std_w = np.array(mean_w), np.nan_to_num(np.array(std_w))
    mean_c1, std_c1 = np.array(mean_c1), np.nan_to_num(np.array(std_c1))
    mean_c2, std_c2 = np.array(mean_c2), np.nan_to_num(np.array(std_c2))

    log_info("Plotting Control Parameters...", module_name)
    plt.figure(figsize=(10, 6))
    title = f"Evaluation: Mean Control Parameters ± Std Dev ({num_runs} Runs)"

    valid_w_mask = ~np.isnan(mean_w)
    if np.any(valid_w_mask):
        plt.plot(valid_steps[valid_w_mask], mean_w[valid_w_mask], label='ω', color='tab:blue', linewidth=1.5)
        if np.any(std_w[valid_w_mask] > 1e-9):
             plt.fill_between(valid_steps[valid_w_mask], mean_w[valid_w_mask] - std_w[valid_w_mask], mean_w[valid_w_mask] + std_w[valid_w_mask], color='tab:blue', alpha=0.2)

    valid_c1_mask = ~np.isnan(mean_c1)
    if np.any(valid_c1_mask):
        plt.plot(valid_steps[valid_c1_mask], mean_c1[valid_c1_mask], label='c1', color='tab:orange', linewidth=1.5)
        if np.any(std_c1[valid_c1_mask] > 1e-9):
             plt.fill_between(valid_steps[valid_c1_mask], mean_c1[valid_c1_mask] - std_c1[valid_c1_mask], mean_c1[valid_c1_mask] + std_c1[valid_c1_mask], color='tab:orange', alpha=0.2)

    valid_c2_mask = ~np.isnan(mean_c2)
    if np.any(valid_c2_mask):
        plt.plot(valid_steps[valid_c2_mask], mean_c2[valid_c2_mask], label='c2', color='tab:red', linewidth=1.5)
        if np.any(std_c2[valid_c2_mask] > 1e-9):
             plt.fill_between(valid_steps[valid_c2_mask], mean_c2[valid_c2_mask] - std_c2[valid_c2_mask], mean_c2[valid_c2_mask] + std_c2[valid_c2_mask], color='tab:red', alpha=0.2)

    plt.xlabel("PSO Time Steps"); plt.ylabel("Parameter Value")
    plt.title(title); plt.legend(loc='best')
    plt.grid(True); plt.ylim(bottom=0); plt.xlim(0, max_steps); plt.tight_layout()
    plot_filename = os.path.join(checkpoint_dir, f"{checkpoint_prefix}_eval_params_mean_std.png")
    try:
        plt.savefig(plot_filename)
        log_success(f"Plot saved: {plot_filename}", module_name)
    except Exception as e:
        log_error(f"Could not save plot {plot_filename}: {e}", module_name)
        log_error(traceback.format_exc(), module_name)
    plt.close(plt.gcf())


def plot_stable_particles(eval_data, max_steps, checkpoint_dir, checkpoint_prefix):
    """Plots mean/std dev for Stable Particle Ratio."""
    metric_index = 3 # Index corresponding to stable_ratio
    metric_name = "Stable Particles Ratio"
    valid_steps, means, stds, num_runs = _aggregate_metric(eval_data, max_steps, metric_index, metric_name)
    # --- Add slight padding to y-limits ---
    _plot_metric_mean_std(valid_steps, means, stds, num_runs,
                         f'Evaluation: Mean {metric_name} ± Std Dev',
                         'Stable Ratio', 'tab:green',
                         checkpoint_dir, f"{checkpoint_prefix}_eval_stable_mean_std.png",
                         max_steps, y_limit_bottom=-0.05, y_limit_top=1.05) # Use slightly padded limits


def plot_infeasible_particles(eval_data, max_steps, checkpoint_dir, checkpoint_prefix):
    """Plots mean/std dev for Infeasible Particle Ratio."""
    metric_index = 4 # Index corresponding to infeasible_ratio
    metric_name = "Infeasible Particles Ratio"
    valid_steps, means, stds, num_runs = _aggregate_metric(eval_data, max_steps, metric_index, metric_name)
    # --- Add slight padding to y-limits ---
    _plot_metric_mean_std(valid_steps, means, stds, num_runs,
                         f'Evaluation: Mean {metric_name} ± Std Dev',
                         'Infeasible Ratio', 'tab:purple',
                         checkpoint_dir, f"{checkpoint_prefix}_eval_infeasible_mean_std.png",
                         max_steps, y_limit_bottom=-0.05, y_limit_top=1.05) # Use slightly padded limits


def plot_average_velocity(eval_data, max_steps, checkpoint_dir, checkpoint_prefix):
    """Plots mean/std dev for Average Particle Velocity Magnitude."""
    # Note: This plots avg_current_velocity_magnitude (index 5)
    metric_index = 5
    metric_name = "Average Velocity Magnitude" # Title reflects what's plotted
    valid_steps, means, stds, num_runs = _aggregate_metric(eval_data, max_steps, metric_index, metric_name)
    _plot_metric_mean_std(valid_steps, means, stds, num_runs,
                         f'Evaluation: Mean {metric_name} ± Std Dev',
                         'Avg Velocity Magnitude', 'tab:cyan',
                         checkpoint_dir, f"{checkpoint_prefix}_eval_avg_vel_mean_std.png",
                         max_steps, use_log_scale=True, y_limit_bottom=1e-6)


def plot_swarm_diversity(eval_data, max_steps, checkpoint_dir, checkpoint_prefix):
    """Plots mean/std dev for Swarm Diversity."""
    metric_index = 6 # Index corresponding to swarm_diversity
    metric_name = "Swarm Diversity"
    valid_steps, means, stds, num_runs = _aggregate_metric(eval_data, max_steps, metric_index, metric_name)
    _plot_metric_mean_std(valid_steps, means, stds, num_runs,
                         f'Evaluation: Mean {metric_name} ± Std Dev',
                         'Swarm Diversity', 'tab:pink',
                         checkpoint_dir, f"{checkpoint_prefix}_eval_diversity_mean_std.png",
                         max_steps, use_log_scale=True, y_limit_bottom=1e-3)


def plot_gbest_convergence(eval_data, max_steps, checkpoint_dir, checkpoint_prefix):
    """Plots mean/std dev for Global Best Value Convergence."""
    metric_index = 7 # Index corresponding to gbest_val
    metric_name = "Global Best Value"
    valid_steps, means, stds, num_runs = _aggregate_metric(eval_data, max_steps, metric_index, metric_name)
    _plot_metric_mean_std(valid_steps, means, stds, num_runs,
                         f'Evaluation: Mean {metric_name} ± Std Dev',
                         'Global Best Value', 'tab:gray',
                         checkpoint_dir, f"{checkpoint_prefix}_eval_gbest_mean_std.png",
                         max_steps, use_log_scale=True, y_limit_bottom=None)

