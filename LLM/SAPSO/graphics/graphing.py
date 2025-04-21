# File: LLM/SAPSO/graphics/graphing.py
# Refactored to use the logger module

import matplotlib.pyplot as plt
import numpy as np
import os
import traceback # For logging exceptions
from pathlib import Path # To get module name

# --- Import Logger ---
# Using the specified import path: from LLM.Logs import logger
# Assuming Logs directory is two levels up from graphics directory
try:
    from ...Logs import logger # Adjust relative path if needed
    from ...Logs.logger import log_info, log_error, log_warning, log_success, log_header, log_debug
except ImportError:
    # Fallback print if logger fails to import
    print("ERROR: Logger module not found at 'LLM.Logs.logger'. Please check path.")
    print("Falling back to standard print statements.")
    # Define dummy functions
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
    log_debug(f"Aggregating '{metric_name}' data...", module_name)
    steps = sorted(eval_data.keys())
    if not steps:
        log_warning(f"No evaluation data steps found for metric '{metric_name}'.", module_name)
        return None, None, None, 0 # Added num_runs return

    # Initialize lists
    metric_values = [[] for _ in range(max_steps)]
    num_runs = 0
    try:
        # Estimate number of runs from the first step with data
        first_step_with_data = next((step for step in steps if eval_data.get(step)), -1)
        if first_step_with_data != -1 and eval_data[first_step_with_data]:
             num_runs = len(eval_data[first_step_with_data])
        else:
             log_warning(f"No steps with actual data found for metric '{metric_name}'. Cannot determine number of runs.", module_name)
             num_runs = 0

    except Exception as e:
        log_warning(f"Could not determine number of runs for metric '{metric_name}': {e}", module_name)
        num_runs = 0


    # Populate lists
    valid_data_points_count = 0
    for step in steps:
        if step < max_steps and step in eval_data: # Check if step exists in dict
            step_data = eval_data[step] # List of data points for this step
            for run_data_point in step_data:
                # Data format: [w, c1, c2, stable%, infeasible%, avg_vel, diversity, gbest]
                if run_data_point is not None and len(run_data_point) > metric_index: # Check if metric exists
                    metric_val = run_data_point[metric_index]
                    if metric_val is not None and np.isfinite(metric_val): # Only add valid numbers
                        metric_values[step].append(metric_val)
                        valid_data_points_count += 1
                    # else: log_debug(f"NaN or None found at step {step} for metric {metric_name}", module_name) # Optional debug
                # else: log_debug(f"Invalid run_data_point or index at step {step} for metric {metric_name}", module_name)

    log_debug(f"Found {valid_data_points_count} valid data points for '{metric_name}'.", module_name)

    # Calculate mean and std dev
    means, stds = [], []
    valid_steps = []

    for i in range(max_steps):
        step_vals = metric_values[i]
        if len(step_vals) > 1: # Need >1 point for std dev
            valid_steps.append(i)
            means.append(np.mean(step_vals))
            stds.append(np.std(step_vals))
        elif len(step_vals) == 1: # Handle single point
             valid_steps.append(i)
             means.append(step_vals[0])
             stds.append(0.0) # Standard deviation is 0 for a single point
        # else: Step 'i' had 0 valid data points, skip

    if not valid_steps:
        log_warning(f"Not enough valid data points across steps/runs for metric '{metric_name}'. Cannot plot.", module_name)
        return None, None, None, num_runs

    # Return aggregated data and the estimated number of runs
    log_debug(f"Aggregation complete for '{metric_name}'. Found {len(valid_steps)} steps with data.", module_name)
    return np.array(valid_steps), np.array(means), np.array(stds), num_runs

# === Generic plotting function ===
def _plot_metric_mean_std(valid_steps, means, stds, num_runs,
                         title_base, ylabel, color,
                         checkpoint_dir, filename_suffix,
                         max_steps, use_log_scale=False,
                         y_limit_bottom=None, y_limit_top=None):
    """Generic function to plot mean and std dev for a metric."""
    if valid_steps is None or len(valid_steps) == 0:
        # Already logged warning in _aggregate_metric
        return

    log_info(f"Generating plot: {title_base}", module_name)
    plt.figure(figsize=(10, 6))
    # Add number of runs to the title for context
    title = f'{title_base} ({num_runs} Runs)'
    plot_label = f'{ylabel}' # Simpler label, title has details

    # Plot mean line
    plt.plot(valid_steps, means, label=plot_label, color=color, linewidth=1.5) # Thinner line
    # Plot shaded standard deviation area
    plt.fill_between(valid_steps, means - stds, means + stds,
                     color=color, alpha=0.2) # No label for shaded area

    plt.xlabel("PSO Time Steps")
    plt.ylabel(ylabel + (" (log scale)" if use_log_scale else ""))
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True, which='both' if use_log_scale else 'major', linestyle='--')

    # --- Y-axis scaling and limits ---
    current_bottom, current_top = plt.ylim() # Get default limits

    if use_log_scale:
        plt.yscale('log')
        # Adjust bottom limit for log scale if necessary
        safe_means = means[means > 0] # Consider only positive means for log scale min
        min_positive_mean = np.min(safe_means) if len(safe_means) > 0 else None

        if y_limit_bottom is None:
            # Set a sensible default bottom limit for log scale
            if min_positive_mean is not None and min_positive_mean > 0:
                 y_limit_bottom = min_positive_mean * 0.1 # Example: one order magnitude lower
            else:
                 y_limit_bottom = 1e-6 # Default small positive value if no positive means
        # Ensure bottom limit is positive for log scale
        y_limit_bottom = max(y_limit_bottom, np.finfo(float).tiny) # Use smallest positive float if needed

    # Apply limits, ensuring bottom < top
    final_bottom = y_limit_bottom if y_limit_bottom is not None else current_bottom
    final_top = y_limit_top if y_limit_top is not None else current_top

    # Prevent invalid limits (bottom >= top)
    if final_bottom >= final_top:
        log_warning(f"Invalid y-limits for plot '{title}': bottom ({final_bottom}) >= top ({final_top}). Adjusting top limit.", module_name)
        final_top = final_bottom * 10 if final_bottom > 0 else 1.0 # Adjust top limit based on bottom

    plt.ylim(bottom=final_bottom, top=final_top)
    # --- End Y-axis scaling ---

    plt.xlim(0, max_steps)
    plt.tight_layout()

    # Save the plot
    os.makedirs(checkpoint_dir, exist_ok=True)
    plot_filename = os.path.join(checkpoint_dir, filename_suffix)
    try:
        plt.savefig(plot_filename)
        log_success(f"Plot saved to {plot_filename}", module_name)
    except Exception as e:
        log_error(f"Could not save plot {plot_filename}: {e}", module_name)
        log_error(traceback.format_exc(), module_name)
    # plt.show() # Keep commented out - typically called by the main script
    plt.close(plt.gcf()) # Close the figure to free memory


# === Specific Plotting Functions ===

def plot_evaluation_parameters(eval_data, max_steps, checkpoint_dir, checkpoint_prefix):
    """Plots mean/std dev for Control Parameters (omega, c1, c2)."""
    log_info("Extracting and plotting Control Parameter data...", module_name)
    steps = sorted(eval_data.keys())
    if not steps:
        log_warning("No evaluation data found for Control Parameter plotting.", module_name)
        return

    # Initialize lists
    omegas = [[] for _ in range(max_steps)]
    c1s    = [[] for _ in range(max_steps)]
    c2s    = [[] for _ in range(max_steps)]
    num_runs = 0
    try: # Estimate number of runs
        first_step_with_data = next((step for step in steps if eval_data.get(step)), -1)
        if first_step_with_data != -1 and eval_data[first_step_with_data]:
             num_runs = len(eval_data[first_step_with_data])
    except Exception as e:
        log_warning(f"Could not determine number of runs for CPs: {e}", module_name)

    # Populate lists
    for step in steps:
        if step < max_steps and step in eval_data:
            for run_data in eval_data[step]:
                # Data format: [w, c1, c2, stable%, infeasible%, avg_vel, diversity, gbest]
                if run_data is not None and len(run_data) >= 3:
                    if run_data[0] is not None and np.isfinite(run_data[0]): omegas[step].append(run_data[0])
                    if run_data[1] is not None and np.isfinite(run_data[1]): c1s[step].append(run_data[1])
                    if run_data[2] is not None and np.isfinite(run_data[2]): c2s[step].append(run_data[2])

    # Aggregate means and stds
    mean_w, std_w, mean_c1, std_c1, mean_c2, std_c2 = [],[],[],[],[],[]
    valid_steps = []
    for i in range(max_steps):
        # Check each parameter list individually for sufficient data
        valid_w = len(omegas[i]) > 1; single_w = len(omegas[i]) == 1
        valid_c1 = len(c1s[i]) > 1;   single_c1 = len(c1s[i]) == 1
        valid_c2 = len(c2s[i]) > 1;   single_c2 = len(c2s[i]) == 1

        if valid_w or valid_c1 or valid_c2 or single_w or single_c1 or single_c2:
             valid_steps.append(i)
             # Calculate mean/std only if data exists for that step, otherwise NaN
             mean_w.append(np.mean(omegas[i]) if (valid_w or single_w) else np.nan)
             std_w.append(np.std(omegas[i]) if valid_w else (0.0 if single_w else np.nan))
             mean_c1.append(np.mean(c1s[i]) if (valid_c1 or single_c1) else np.nan)
             std_c1.append(np.std(c1s[i]) if valid_c1 else (0.0 if single_c1 else np.nan))
             mean_c2.append(np.mean(c2s[i]) if (valid_c2 or single_c2) else np.nan)
             std_c2.append(np.std(c2s[i]) if valid_c2 else (0.0 if single_c2 else np.nan))


    if not valid_steps:
        log_warning("Not enough valid data found for Control Parameter plot.", module_name)
        return

    valid_steps = np.array(valid_steps)
    # Convert lists to numpy arrays for plotting, handling potential NaNs
    mean_w, std_w = np.array(mean_w), np.nan_to_num(np.array(std_w)) # Replace NaN std with 0
    mean_c1, std_c1 = np.array(mean_c1), np.nan_to_num(np.array(std_c1))
    mean_c2, std_c2 = np.array(mean_c2), np.nan_to_num(np.array(std_c2))

    log_info("Plotting Control Parameters...", module_name)
    plt.figure(figsize=(10, 6))
    title = f"Evaluation: Mean Control Parameters ± Std Dev ({num_runs} Runs)"

    # Plot Omega (w) - Blue, only where mean is not NaN
    valid_w_mask = ~np.isnan(mean_w)
    if np.any(valid_w_mask):
        plt.plot(valid_steps[valid_w_mask], mean_w[valid_w_mask], label='ω', color='tab:blue', linewidth=1.5)
        plt.fill_between(valid_steps[valid_w_mask], mean_w[valid_w_mask] - std_w[valid_w_mask], mean_w[valid_w_mask] + std_w[valid_w_mask], color='tab:blue', alpha=0.2)

    # Plot c1 - Orange, only where mean is not NaN
    valid_c1_mask = ~np.isnan(mean_c1)
    if np.any(valid_c1_mask):
        plt.plot(valid_steps[valid_c1_mask], mean_c1[valid_c1_mask], label='c1', color='tab:orange', linewidth=1.5)
        plt.fill_between(valid_steps[valid_c1_mask], mean_c1[valid_c1_mask] - std_c1[valid_c1_mask], mean_c1[valid_c1_mask] + std_c1[valid_c1_mask], color='tab:orange', alpha=0.2)

    # Plot c2 - Red, only where mean is not NaN
    valid_c2_mask = ~np.isnan(mean_c2)
    if np.any(valid_c2_mask):
        plt.plot(valid_steps[valid_c2_mask], mean_c2[valid_c2_mask], label='c2', color='tab:red', linewidth=1.5)
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
    # plt.show() # Keep commented out
    plt.close(plt.gcf()) # Close figure


def plot_stable_particles(eval_data, max_steps, checkpoint_dir, checkpoint_prefix):
    """Plots mean/std dev for Stable Particle Ratio."""
    metric_index = 3 # Index corresponding to stable_ratio
    metric_name = "Stable Particles Ratio"
    valid_steps, means, stds, num_runs = _aggregate_metric(eval_data, max_steps, metric_index, metric_name)
    _plot_metric_mean_std(valid_steps, means, stds, num_runs,
                         f'Evaluation: Mean {metric_name} ± Std Dev',
                         'Stable Ratio', 'tab:green',
                         checkpoint_dir, f"{checkpoint_prefix}_eval_stable_mean_std.png",
                         max_steps, y_limit_bottom=0.0, y_limit_top=1.0)


def plot_infeasible_particles(eval_data, max_steps, checkpoint_dir, checkpoint_prefix):
    """Plots mean/std dev for Infeasible Particle Ratio."""
    metric_index = 4 # Index corresponding to infeasible_ratio
    metric_name = "Infeasible Particles Ratio"
    valid_steps, means, stds, num_runs = _aggregate_metric(eval_data, max_steps, metric_index, metric_name)
    _plot_metric_mean_std(valid_steps, means, stds, num_runs,
                         f'Evaluation: Mean {metric_name} ± Std Dev',
                         'Infeasible Ratio', 'tab:purple',
                         checkpoint_dir, f"{checkpoint_prefix}_eval_infeasible_mean_std.png",
                         max_steps, y_limit_bottom=0.0, y_limit_top=1.0)


def plot_average_velocity(eval_data, max_steps, checkpoint_dir, checkpoint_prefix):
    """Plots mean/std dev for Average Particle Velocity Magnitude."""
    metric_index = 5 # Index corresponding to avg_vel_mag
    metric_name = "Average Velocity Magnitude"
    valid_steps, means, stds, num_runs = _aggregate_metric(eval_data, max_steps, metric_index, metric_name)
    _plot_metric_mean_std(valid_steps, means, stds, num_runs,
                         f'Evaluation: Mean {metric_name} ± Std Dev',
                         'Avg Velocity Magnitude', 'tab:cyan',
                         checkpoint_dir, f"{checkpoint_prefix}_eval_avg_vel_mean_std.png",
                         max_steps, use_log_scale=True, y_limit_bottom=1e-6) # Use log scale


def plot_swarm_diversity(eval_data, max_steps, checkpoint_dir, checkpoint_prefix):
    """Plots mean/std dev for Swarm Diversity."""
    metric_index = 6 # Index corresponding to diversity
    metric_name = "Swarm Diversity"
    valid_steps, means, stds, num_runs = _aggregate_metric(eval_data, max_steps, metric_index, metric_name)
    _plot_metric_mean_std(valid_steps, means, stds, num_runs,
                         f'Evaluation: Mean {metric_name} ± Std Dev',
                         'Swarm Diversity', 'tab:pink',
                         checkpoint_dir, f"{checkpoint_prefix}_eval_diversity_mean_std.png",
                         max_steps, use_log_scale=True, y_limit_bottom=1e-3) # Use log scale


def plot_gbest_convergence(eval_data, max_steps, checkpoint_dir, checkpoint_prefix):
    """Plots mean/std dev for Global Best Value Convergence."""
    metric_index = 7 # Index corresponding to gbest_val
    metric_name = "Global Best Value"
    valid_steps, means, stds, num_runs = _aggregate_metric(eval_data, max_steps, metric_index, metric_name)
    _plot_metric_mean_std(valid_steps, means, stds, num_runs,
                         f'Evaluation: Mean {metric_name} ± Std Dev',
                         'Global Best Value', 'tab:gray',
                         checkpoint_dir, f"{checkpoint_prefix}_eval_gbest_mean_std.png",
                         max_steps, use_log_scale=True, y_limit_bottom=None) # Let bottom limit be determined automatically for gbest

