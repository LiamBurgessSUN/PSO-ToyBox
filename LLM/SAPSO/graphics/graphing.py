import matplotlib.pyplot as plt
import numpy as np
import os

# === Helper function to aggregate data ===
def _aggregate_metric(eval_data, max_steps, metric_index, metric_name):
    """Helper to extract and aggregate a specific metric from eval_data."""
    print(f"Aggregating '{metric_name}' data...")
    steps = sorted(eval_data.keys())
    if not steps:
        print(f"No evaluation data for {metric_name}.")
        return None, None, None, 0 # Added num_runs return

    # Initialize lists
    metric_values = [[] for _ in range(max_steps)]
    num_runs = 0
    try:
        # Estimate number of runs from the first step with data
        first_step_with_data = next(step for step in steps if eval_data[step])
        num_runs = len(eval_data[first_step_with_data])
    except StopIteration:
        print(f"Warning: No steps with data found for {metric_name}.")
        num_runs = 0


    # Populate lists
    for step in steps:
        if step < max_steps:
            step_data = eval_data[step] # List of data points for this step
            for run_data_point in step_data:
                if len(run_data_point) > metric_index: # Check if metric exists
                    metric_val = run_data_point[metric_index]
                    if metric_val is not None and not np.isnan(metric_val): # Only add valid numbers
                        metric_values[step].append(metric_val)
                    # else: print(f"Debug: NaN or None found at step {step} for metric {metric_name}") # Optional debug

    # Calculate mean and std dev
    means, stds = [], []
    valid_steps = []

    for i in range(max_steps):
        if len(metric_values[i]) > 1: # Need >1 point for std dev
            valid_steps.append(i)
            means.append(np.mean(metric_values[i]))
            stds.append(np.std(metric_values[i]))
        elif len(metric_values[i]) == 1: # Handle single point
             valid_steps.append(i)
             means.append(metric_values[i][0])
             stds.append(0)
        # else: Step 'i' had 0 or 1 valid data point, skip

    if not valid_steps:
        print(f"Not enough valid data points across multiple runs for {metric_name}.")
        return None, None, None, num_runs

    # Return aggregated data and the estimated number of runs
    return np.array(valid_steps), np.array(means), np.array(stds), num_runs

# === Generic plotting function ===
def _plot_metric_mean_std(valid_steps, means, stds, num_runs,
                         title_base, ylabel, color,
                         checkpoint_dir, filename_suffix,
                         max_steps, use_log_scale=False,
                         y_limit_bottom=None, y_limit_top=None):
    """Generic function to plot mean and std dev for a metric."""
    if valid_steps is None or len(valid_steps) == 0:
        print(f"Skipping plot '{title_base}': No valid data.")
        return

    plt.figure(figsize=(10, 6))
    title = f'{title_base} ({num_runs} Runs)'
    plot_label = f'{ylabel}' # Simpler label

    # Plot mean line
    plt.plot(valid_steps, means, label=plot_label, color=color, linewidth=2)
    # Plot shaded standard deviation area
    plt.fill_between(valid_steps, means - stds, means + stds,
                     color=color, alpha=0.2) # No label for shaded area

    plt.xlabel("PSO Time Steps")
    plt.ylabel(ylabel + (" (log scale)" if use_log_scale else ""))
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True, which='both' if use_log_scale else 'major', linestyle='--')

    if use_log_scale:
        plt.yscale('log')
        # Adjust bottom limit for log scale if necessary
        if y_limit_bottom is None and len(means)>0:
             min_val = np.min(means[means > 0]) # Find min positive value for sensible limit
             if min_val > 0:
                 y_limit_bottom = min_val * 0.1 # Example: 1 order magnitude lower
             else:
                 y_limit_bottom = 1e-6 # Default small value if no positive mean found

    if y_limit_bottom is not None:
        plt.ylim(bottom=y_limit_bottom)
    if y_limit_top is not None:
        # Ensure top limit is greater than bottom limit, esp. for log scale
        if y_limit_bottom is None or y_limit_top > y_limit_bottom:
            plt.ylim(top=y_limit_top)
        else:
             print(f"Warning: Invalid y_limit_top ({y_limit_top}) <= y_limit_bottom ({y_limit_bottom}) for plot '{title}'. Adjusting.")
             plt.ylim(top=y_limit_bottom * 10) # Example adjustment


    plt.xlim(0, max_steps)
    plt.tight_layout()

    # Save the plot
    os.makedirs(checkpoint_dir, exist_ok=True)
    plot_filename = os.path.join(checkpoint_dir, filename_suffix)
    try:
        plt.savefig(plot_filename)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Warning: Could not save plot {plot_filename}: {e}")
    plt.show()


# === Specific Plotting Functions ===

def plot_evaluation_parameters(eval_data, max_steps, checkpoint_dir, checkpoint_prefix):
    """Plots mean/std dev for Control Parameters (omega, c1, c2)."""
    print("Extracting Control Parameter data...")
    steps = sorted(eval_data.keys())
    if not steps: print("No evaluation data for CP plotting."); return

    # Initialize lists
    omegas = [[] for _ in range(max_steps)]
    c1s    = [[] for _ in range(max_steps)]
    c2s    = [[] for _ in range(max_steps)]
    num_runs = 0
    try:
        first_step_with_data = next(step for step in steps if eval_data[step])
        num_runs = len(eval_data[first_step_with_data])
    except StopIteration: num_runs = 0

    # Populate lists
    for step in steps:
        if step < max_steps:
            for run_data in eval_data[step]:
                # Data format: [w, c1, c2, stable%, infeasible%, avg_vel, diversity, gbest]
                if len(run_data) >= 3:
                    if run_data[0] is not None and not np.isnan(run_data[0]): omegas[step].append(run_data[0])
                    if run_data[1] is not None and not np.isnan(run_data[1]): c1s[step].append(run_data[1])
                    if run_data[2] is not None and not np.isnan(run_data[2]): c2s[step].append(run_data[2])

    # Aggregate means and stds
    mean_w, std_w, mean_c1, std_c1, mean_c2, std_c2 = [],[],[],[],[],[]
    valid_steps = []
    for i in range(max_steps):
        # Check each parameter list individually for sufficient data
        valid_w = len(omegas[i]) > 1
        valid_c1 = len(c1s[i]) > 1
        valid_c2 = len(c2s[i]) > 1
        single_w = len(omegas[i]) == 1
        single_c1 = len(c1s[i]) == 1
        single_c2 = len(c2s[i]) == 1

        if valid_w or valid_c1 or valid_c2 or single_w or single_c1 or single_c2:
             valid_steps.append(i)
             mean_w.append(np.mean(omegas[i]) if (valid_w or single_w) else np.nan)
             std_w.append(np.std(omegas[i]) if valid_w else (0 if single_w else np.nan))
             mean_c1.append(np.mean(c1s[i]) if (valid_c1 or single_c1) else np.nan)
             std_c1.append(np.std(c1s[i]) if valid_c1 else (0 if single_c1 else np.nan))
             mean_c2.append(np.mean(c2s[i]) if (valid_c2 or single_c2) else np.nan)
             std_c2.append(np.std(c2s[i]) if valid_c2 else (0 if single_c2 else np.nan))


    if not valid_steps: print("Not enough data for Control Parameter plot."); return

    valid_steps = np.array(valid_steps)
    mean_w, std_w = np.array(mean_w), np.array(std_w)
    mean_c1, std_c1 = np.array(mean_c1), np.array(std_c1)
    mean_c2, std_c2 = np.array(mean_c2), np.array(std_c2)

    print("Plotting Control Parameters...")
    plt.figure(figsize=(10, 6))
    # Plot Omega (w) - Blue
    plt.plot(valid_steps, mean_w, label='ω', color='tab:blue', linewidth=2)
    plt.fill_between(valid_steps, mean_w - std_w, mean_w + std_w, color='tab:blue', alpha=0.2)
    # Plot c1 - Orange
    plt.plot(valid_steps, mean_c1, label='c1', color='tab:orange', linewidth=2)
    plt.fill_between(valid_steps, mean_c1 - std_c1, mean_c1 + std_c1, color='tab:orange', alpha=0.2)
    # Plot c2 - Red
    plt.plot(valid_steps, mean_c2, label='c2', color='tab:red', linewidth=2)
    plt.fill_between(valid_steps, mean_c2 - std_c2, mean_c2 + std_c2, color='tab:red', alpha=0.2)

    plt.xlabel("PSO Time Steps"); plt.ylabel("Parameter Value")
    plt.title(f"Evaluation: Mean Control Parameters ± Std Dev ({num_runs} Runs)"); plt.legend(loc='best')
    plt.grid(True); plt.ylim(bottom=0); plt.xlim(0, max_steps); plt.tight_layout()
    plot_filename = os.path.join(checkpoint_dir, f"{checkpoint_prefix}_eval_params_mean_std.png")
    try: plt.savefig(plot_filename); print(f"Saved plot: {plot_filename}")
    except Exception as e: print(f"Warning: Could not save plot {plot_filename}: {e}")
    plt.show()


def plot_stable_particles(eval_data, max_steps, checkpoint_dir, checkpoint_prefix):
    """Plots mean/std dev for Stable Particle Ratio."""
    metric_index = 3 # Index corresponding to stable_ratio
    valid_steps, means, stds, num_runs = _aggregate_metric(eval_data, max_steps, metric_index, "Stable Particles Ratio")
    _plot_metric_mean_std(valid_steps, means, stds, num_runs,
                         'Evaluation: Mean Stable Particle Ratio ± Std Dev',
                         'Stable Ratio', 'tab:green',
                         checkpoint_dir, f"{checkpoint_prefix}_eval_stable_mean_std.png",
                         max_steps, y_limit_bottom=0.0, y_limit_top=1.0)


def plot_infeasible_particles(eval_data, max_steps, checkpoint_dir, checkpoint_prefix):
    """Plots mean/std dev for Infeasible Particle Ratio."""
    metric_index = 4 # Index corresponding to infeasible_ratio
    valid_steps, means, stds, num_runs = _aggregate_metric(eval_data, max_steps, metric_index, "Infeasible Particles Ratio")
    _plot_metric_mean_std(valid_steps, means, stds, num_runs,
                         'Evaluation: Mean Infeasible Particle Ratio ± Std Dev',
                         'Infeasible Ratio', 'tab:purple',
                         checkpoint_dir, f"{checkpoint_prefix}_eval_infeasible_mean_std.png",
                         max_steps, y_limit_bottom=0.0, y_limit_top=1.0)


def plot_average_velocity(eval_data, max_steps, checkpoint_dir, checkpoint_prefix):
    """Plots mean/std dev for Average Particle Velocity Magnitude."""
    metric_index = 5 # Index corresponding to avg_vel_mag
    valid_steps, means, stds, num_runs = _aggregate_metric(eval_data, max_steps, metric_index, "Average Velocity Magnitude")
    _plot_metric_mean_std(valid_steps, means, stds, num_runs,
                         'Evaluation: Mean Average Particle Velocity Mag ± Std Dev',
                         'Avg Velocity Magnitude', 'tab:cyan',
                         checkpoint_dir, f"{checkpoint_prefix}_eval_avg_vel_mean_std.png",
                         max_steps, use_log_scale=True, y_limit_bottom=1e-6) # Use log scale


def plot_swarm_diversity(eval_data, max_steps, checkpoint_dir, checkpoint_prefix):
    """Plots mean/std dev for Swarm Diversity."""
    metric_index = 6 # Index corresponding to diversity
    valid_steps, means, stds, num_runs = _aggregate_metric(eval_data, max_steps, metric_index, "Swarm Diversity")
    _plot_metric_mean_std(valid_steps, means, stds, num_runs,
                         'Evaluation: Mean Swarm Diversity ± Std Dev',
                         'Swarm Diversity', 'tab:pink', # Changed color
                         checkpoint_dir, f"{checkpoint_prefix}_eval_diversity_mean_std.png",
                         max_steps, use_log_scale=True, y_limit_bottom=1e-3) # Use log scale


def plot_gbest_convergence(eval_data, max_steps, checkpoint_dir, checkpoint_prefix):
    """Plots mean/std dev for Global Best Value Convergence."""
    metric_index = 7 # Index corresponding to gbest_val
    valid_steps, means, stds, num_runs = _aggregate_metric(eval_data, max_steps, metric_index, "Global Best Value")
    _plot_metric_mean_std(valid_steps, means, stds, num_runs,
                         'Evaluation: Mean Global Best Value ± Std Dev',
                         'Global Best Value', 'tab:gray',
                         checkpoint_dir, f"{checkpoint_prefix}_eval_gbest_mean_std.png",
                         max_steps, use_log_scale=True, y_limit_bottom=1e-6) # Use log scale by default

