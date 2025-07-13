# File: SAPSO_AGENT/SAPSO/Graphics/sapso_plotting.py
# Enhanced plotting functions for SAPSO training loop, based on baseline PSO plotting
# Integrates with the existing SwarmMetrics class to provide comprehensive plotting capabilities

import matplotlib.pyplot as plt
import numpy as np
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime

from SAPSO_AGENT.Logs.logger import *
from SAPSO_AGENT.CONFIG import ENV_MAX_STEPS

# --- Module Name for Logging ---
module_name = Path(__file__).stem  # Gets 'sapso_plotting'


def generate_timestamped_filename(base_name: str, extension: str = "png") -> str:
    """
    Generate a filename with timestamp and base name.
    
    Args:
        base_name: The base name for the file
        extension: File extension (default: "png")
    
    Returns:
        str: Timestamped filename in format "YYYYMMDD_HHMMSS_base_name.extension"
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{base_name}.{extension}"


class SAPSOPlotter:
    """
    Enhanced plotting class for SAPSO training loop, based on baseline PSO plotting functions.
    Integrates with SwarmMetrics to provide comprehensive plotting capabilities.
    """
    
    def __init__(self, checkpoint_base_dir: Optional[str] = None, plot_only_averages: bool = False):
        """
        Initialize the SAPSO plotter.
        
        Args:
            checkpoint_base_dir: Base directory for saving plots
            plot_only_averages: If True, only generate average plots, skip individual function plots
        """
        self.checkpoint_base_dir = checkpoint_base_dir
        self.plot_only_averages = plot_only_averages
        self.module_name = "SAPSOPlotter"
        
    def plot_parameter_evolution(self, 
                                metrics_calculator,
                                function_names: Optional[List[str]] = None,
                                save_plots: bool = True,
                                show_plots: bool = False) -> None:
        """
        Plot the average control parameter values (omega, c1, c2) over the optimization steps.
        
        Args:
            metrics_calculator: SwarmMetrics instance with tracking data
            function_names: List of function names to plot. If None, plots all tracked functions.
            save_plots: Whether to save the plots to files
            show_plots: Whether to display the plots
        """
        if not hasattr(metrics_calculator, 'parameter_tracking') or not metrics_calculator.parameter_tracking:
            log_warning("No parameter tracking data available. Run PSO evaluation first.", self.module_name)
            return
        
        if function_names is None:
            function_names = list(metrics_calculator.parameter_tracking.keys())
        
        # Skip individual function plots if plot_only_averages is True
        if self.plot_only_averages:
            log_info("Skipping individual function plots (plot_only_averages=True)", self.module_name)
            return
        
        log_info(f"Plotting parameter evolution for {len(function_names)} functions", self.module_name)
        
        # Create figure with subplots for each function
        num_functions = len(function_names)
        fig, axes = plt.subplots(num_functions, 1, figsize=(12, 4 * num_functions))
        
        # Ensure axes is always a list
        if num_functions == 1:
            axes = [axes]
        
        for i, func_name in enumerate(function_names):
            if func_name not in metrics_calculator.parameter_tracking:
                log_warning(f"No parameter data for function {func_name}", self.module_name)
                continue
            
            tracking_data = metrics_calculator.parameter_tracking[func_name]
            
            # Get parameter values
            omega_values = np.array(tracking_data['omega'])
            c1_values = np.array(tracking_data['c1'])
            c2_values = np.array(tracking_data['c2'])
            
            steps = np.arange(len(omega_values))
            
            ax = axes[i]
            
            # Plot parameter values
            ax.plot(steps, omega_values, 'b-', label='ω (Inertia)', linewidth=2)
            ax.plot(steps, c1_values, 'r-', label='c₁ (Cognitive)', linewidth=2)
            ax.plot(steps, c2_values, 'g-', label='c₂ (Social)', linewidth=2)
            
            ax.set_xlabel('Optimization Step')
            ax.set_ylabel('Parameter Value')
            ax.set_title(f'Parameter Evolution: {func_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Set y-axis limits to show the full range
            all_values = np.concatenate([omega_values, c1_values, c2_values])
            y_min, y_max = np.min(all_values), np.max(all_values)
            y_range = y_max - y_min
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plots and self.checkpoint_base_dir is not None:
            checkpoint_dir = Path(self.checkpoint_base_dir) / "parameter_evolution"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Create filename with function names
            func_suffix = "_".join([name.replace("Function", "") for name in function_names[:3]])
            if len(function_names) > 3:
                func_suffix += f"_and_{len(function_names)-3}_more"
            
            timestamped_filename = generate_timestamped_filename(f"sapso_parameter_evolution_{func_suffix}")
            plot_path = checkpoint_dir / timestamped_filename
            
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            log_success(f"Parameter evolution plot saved to {plot_path}", self.module_name)
        
        # Show plot if requested
        if show_plots:
            plt.show()
        
        plt.close()
    
    def plot_parameter_comparison(self, 
                                 metrics_calculator,
                                 function_names: Optional[List[str]] = None,
                                 save_plots: bool = True,
                                 show_plots: bool = False) -> None:
        """
        Create a comparison plot showing parameter evolution across multiple functions.
        
        Args:
            metrics_calculator: SwarmMetrics instance with tracking data
            function_names: List of function names to plot. If None, plots all tracked functions.
            save_plots: Whether to save the plots to files
            show_plots: Whether to display the plots
        """
        if not hasattr(metrics_calculator, 'parameter_tracking') or not metrics_calculator.parameter_tracking:
            log_warning("No parameter tracking data available. Run PSO evaluation first.", self.module_name)
            return
        
        if function_names is None:
            function_names = list(metrics_calculator.parameter_tracking.keys())
        
        # Skip individual function plots if plot_only_averages is True
        if self.plot_only_averages:
            log_info("Skipping parameter comparison plots (plot_only_averages=True)", self.module_name)
            return
        
        log_info(f"Creating parameter comparison plot for {len(function_names)} functions", self.module_name)
        
        # Create figure with subplots for each parameter
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Use a simple color palette
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for param_idx, (param_name, param_key) in enumerate([('ω (Inertia)', 'omega'), 
                                                            ('c₁ (Cognitive)', 'c1'), 
                                                            ('c₂ (Social)', 'c2')]):
            ax = axes[param_idx]
            
            for i, func_name in enumerate(function_names):
                if func_name not in metrics_calculator.parameter_tracking:
                    continue
                
                tracking_data = metrics_calculator.parameter_tracking[func_name]
                param_values = np.array(tracking_data[param_key])
                steps = np.arange(len(param_values))
                
                # Shorten function name for legend
                short_name = func_name.replace("Function", "")
                ax.plot(steps, param_values, color=colors[i], label=short_name, linewidth=1.5)
            
            ax.set_xlabel('Optimization Step')
            ax.set_ylabel('Parameter Value')
            ax.set_title(f'{param_name} Evolution Across Functions')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plots and self.checkpoint_base_dir is not None:
            checkpoint_dir = Path(self.checkpoint_base_dir) / "parameter_comparison"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            timestamped_filename = generate_timestamped_filename("sapso_parameter_comparison")
            plot_path = checkpoint_dir / timestamped_filename
            
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            log_success(f"Parameter comparison plot saved to {plot_path}", self.module_name)
        
        # Show plot if requested
        if show_plots:
            plt.show()
        
        plt.close()
    
    def _infer_max_steps(self, metrics_calculator, function_names, key):
        # Try to infer max steps from the data, fallback to ENV_MAX_STEPS
        max_len = 0
        for func_name in function_names:
            if func_name in metrics_calculator.metric_tracking and key in metrics_calculator.metric_tracking[func_name]:
                arr = metrics_calculator.metric_tracking[func_name][key]
                if len(arr) > max_len:
                    max_len = len(arr)
        return max_len if max_len > 0 else ENV_MAX_STEPS

    def _reshape_and_average(self, arr, step_size):
        arr = np.array(arr)
        usable_len = (len(arr) // step_size) * step_size
        arr = arr[:usable_len]
        if usable_len == 0:
            return np.full(step_size, np.nan), np.full(step_size, 0)
        arr = arr.reshape((-1, step_size))
        avg = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        return avg, std

    def plot_average_velocity(self, metrics_calculator, function_names=None, save_plots=True, show_plots=False):
        if function_names is None:
            function_names = list(metrics_calculator.metric_tracking.keys())
        step_size = ENV_MAX_STEPS
        all_arrs = []
        for func_name in function_names:
            if func_name in metrics_calculator.metric_tracking and 'avg_step_size' in metrics_calculator.metric_tracking[func_name]:
                arr = metrics_calculator.metric_tracking[func_name]['avg_step_size']
                all_arrs.append(arr)
        if not all_arrs:
            log_warning('No velocity data for averaging', self.module_name)
            return
        # Concatenate all runs, then average per step
        all_arrs = np.concatenate(all_arrs)
        avg_per_step, std_per_step = self._reshape_and_average(all_arrs, step_size)
        steps = np.arange(step_size)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.semilogy(steps, avg_per_step, label='Average Velocity Magnitude', color='blue', linewidth=2)
        ax.fill_between(steps, avg_per_step-std_per_step, avg_per_step+std_per_step, color='blue', alpha=0.2, label='Std Dev')
        ax.set_title('Average Particle Velocity (Averaged Across All Functions)')
        ax.set_xlabel('PSO Step')
        ax.set_ylabel('Average Velocity Magnitude (log scale)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        final_velocity = avg_per_step[-1] if len(avg_per_step) > 0 else 0
        mean_velocity = np.nanmean(avg_per_step) if len(avg_per_step) > 0 else 0
        max_velocity = np.nanmax(avg_per_step) if len(avg_per_step) > 0 else 0
        min_velocity = np.nanmin(avg_per_step) if len(avg_per_step) > 0 else 0
        stats_text = f'Final Velocity: {final_velocity:.2e}\nMean Velocity: {mean_velocity:.2e}\nMax Velocity: {max_velocity:.2e}\nMin Velocity: {min_velocity:.2e}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        plt.tight_layout()
        if save_plots and self.checkpoint_base_dir is not None:
            checkpoint_dir = Path(self.checkpoint_base_dir) / 'average_velocity'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            timestamped_filename = generate_timestamped_filename('average_velocity_average')
            plt.savefig(checkpoint_dir / timestamped_filename, dpi=300, bbox_inches='tight')
            log_success(f'Average velocity plot saved: {timestamped_filename}', self.module_name)
        if show_plots:
            plt.show()
        plt.close()

    def plot_swarm_diversity(self, metrics_calculator, function_names=None, save_plots=True, show_plots=False, clip_std_dev=False):
        if function_names is None:
            function_names = list(metrics_calculator.metric_tracking.keys())
        step_size = ENV_MAX_STEPS
        all_arrs = []
        for func_name in function_names:
            if func_name in metrics_calculator.metric_tracking and 'swarm_diversity' in metrics_calculator.metric_tracking[func_name]:
                arr = metrics_calculator.metric_tracking[func_name]['swarm_diversity']
                all_arrs.append(arr)
        if not all_arrs:
            log_warning('No diversity data for averaging', self.module_name)
            return
        all_arrs = np.concatenate(all_arrs)
        avg_per_step, std_per_step = self._reshape_and_average(all_arrs, step_size)
        steps = np.arange(step_size)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.semilogy(steps, avg_per_step, label='Swarm Diversity', color='green', linewidth=2)
        ax.fill_between(steps, avg_per_step-std_per_step, avg_per_step+std_per_step, color='green', alpha=0.2, label='Std Dev')
        ax.set_title('Swarm Diversity (Averaged Across All Functions)')
        ax.set_xlabel('PSO Step')
        ax.set_ylabel('Swarm Diversity (log scale)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        final_diversity = avg_per_step[-1] if len(avg_per_step) > 0 else 0
        mean_diversity = np.nanmean(avg_per_step) if len(avg_per_step) > 0 else 0
        max_diversity = np.nanmax(avg_per_step) if len(avg_per_step) > 0 else 0
        min_diversity = np.nanmin(avg_per_step) if len(avg_per_step) > 0 else 0
        stats_text = f'Final Diversity: {final_diversity:.4f}\nMean Diversity: {mean_diversity:.4f}\nMax Diversity: {max_diversity:.4f}\nMin Diversity: {min_diversity:.4f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        plt.tight_layout()
        if save_plots and self.checkpoint_base_dir is not None:
            checkpoint_dir = Path(self.checkpoint_base_dir) / 'swarm_diversity'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            timestamped_filename = generate_timestamped_filename('swarm_diversity_average')
            plt.savefig(checkpoint_dir / timestamped_filename, dpi=300, bbox_inches='tight')
            log_success(f'Swarm diversity plot saved: {timestamped_filename}', self.module_name)
        if show_plots:
            plt.show()
        plt.close()

    def plot_average_parameters(self, metrics_calculator, function_names=None, save_plots=True, show_plots=False):
        if function_names is None:
            function_names = list(metrics_calculator.parameter_tracking.keys())
        step_size = ENV_MAX_STEPS
        all_omega, all_c1, all_c2 = [], [], []
        for func_name in function_names:
            if func_name in metrics_calculator.parameter_tracking:
                all_omega.extend(metrics_calculator.parameter_tracking[func_name]['omega'])
                all_c1.extend(metrics_calculator.parameter_tracking[func_name]['c1'])
                all_c2.extend(metrics_calculator.parameter_tracking[func_name]['c2'])
        avg_omega, _ = self._reshape_and_average(all_omega, step_size)
        avg_c1, _ = self._reshape_and_average(all_c1, step_size)
        avg_c2, _ = self._reshape_and_average(all_c2, step_size)
        steps = np.arange(step_size)
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot(steps, avg_omega, label='ω (Inertia)', color='blue')
        ax.plot(steps, avg_c1, label='c₁ (Cognitive)', color='red')
        ax.plot(steps, avg_c2, label='c₂ (Social)', color='green')
        ax.set_xlabel('PSO Step')
        ax.set_ylabel('Average Parameter Value')
        ax.set_title('Average Control Parameters Across All Functions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_plots and self.checkpoint_base_dir is not None:
            checkpoint_dir = Path(self.checkpoint_base_dir) / 'average_parameters'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            timestamped_filename = generate_timestamped_filename(f'sapso_average_parameters_{len(function_names)}_functions')
            plot_path = checkpoint_dir / timestamped_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            log_success(f'Average parameter plot saved to {plot_path}', self.module_name)
        if show_plots:
            plt.show()
        plt.close()

    def plot_stability_condition(self, metrics_calculator, function_names=None, save_plots=True, show_plots=False):
        if function_names is None:
            function_names = list(metrics_calculator.parameter_tracking.keys())
        step_size = ENV_MAX_STEPS
        all_omega, all_c1, all_c2 = [], [], []
        for func_name in function_names:
            if func_name in metrics_calculator.parameter_tracking:
                all_omega.extend(metrics_calculator.parameter_tracking[func_name]['omega'])
                all_c1.extend(metrics_calculator.parameter_tracking[func_name]['c1'])
                all_c2.extend(metrics_calculator.parameter_tracking[func_name]['c2'])
        avg_omega, _ = self._reshape_and_average(all_omega, step_size)
        avg_c1, _ = self._reshape_and_average(all_c1, step_size)
        avg_c2, _ = self._reshape_and_average(all_c2, step_size)
        lhs = avg_c1 + avg_c2
        denominator = 7.0 - 5.0 * avg_omega
        rhs = np.where((-1.0 <= avg_omega) & (avg_omega <= 1.0) & (~np.isclose(denominator, 0)), 24.0 * (1.0 - avg_omega**2) / denominator, np.nan)
        is_stable = np.where((-1.0 <= avg_omega) & (avg_omega <= 1.0) & (~np.isclose(denominator, 0)), (lhs < rhs).astype(float), 0.0)
        steps = np.arange(step_size)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        ax1.plot(steps, lhs, label=r"$c_1 + c_2$", color='blue', linewidth=2)
        ax1.plot(steps, rhs, label=r"$\frac{24(1-w^2)}{7-5w}$", color='orange', linewidth=2)
        ax1.set_title('Stability Condition (Averaged Across All Functions)')
        ax1.set_xlabel('PSO Step')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.fill_between(steps, 0, rhs, alpha=0.2, color='green', label='Stable Region')
        ax2.plot(steps, is_stable, label='Fraction of Stable Particles', color='red', linewidth=2)
        ax2.set_title('Fraction of Stable Particles (Averaged Across All Functions)')
        ax2.set_xlabel('PSO Step')
        ax2.set_ylabel('Stability Fraction')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        final_stability = is_stable[-1] if len(is_stable) > 0 else 0
        stats_text = f'Final Stability: {final_stability:.3f}\nMean Stability: {np.nanmean(is_stable):.3f}'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.tight_layout()
        if save_plots and self.checkpoint_base_dir is not None:
            checkpoint_dir = Path(self.checkpoint_base_dir) / 'stability_condition'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            timestamped_filename = generate_timestamped_filename('stability_condition_average')
            plt.savefig(checkpoint_dir / timestamped_filename, dpi=300, bbox_inches='tight')
            log_success(f'Stability condition plot saved: {timestamped_filename}', self.module_name)
        if show_plots:
            plt.show()
        plt.close()
    
    def plot_infeasible_particles(self, 
                                 metrics_calculator,
                                 function_names: Optional[List[str]] = None,
                                 save_plots: bool = True,
                                 show_plots: bool = False) -> None:
        """
        Plot the fraction of infeasible particles over optimization steps for each function.
        
        Args:
            metrics_calculator: SwarmMetrics instance with tracking data
            function_names: List of function names to plot. If None, plots all tracked functions.
            save_plots: Whether to save the plots to files
            show_plots: Whether to display the plots
        """
        if not hasattr(metrics_calculator, 'metric_tracking') or not metrics_calculator.metric_tracking:
            log_warning("No metric tracking data available. Run PSO evaluation first.", self.module_name)
            return

        if function_names is None:
            function_names = list(metrics_calculator.metric_tracking.keys())

        # Skip individual function plots if plot_only_averages is True
        if not self.plot_only_averages:
            log_info(f"Creating infeasible particles plots for {len(function_names)} functions", self.module_name)

            # Per-function plots
            for func_name in function_names:
                if func_name not in metrics_calculator.metric_tracking:
                    log_warning(f"No metric data for function {func_name}", self.module_name)
                    continue

                data = metrics_calculator.metric_tracking[func_name]
                
                # Check if infeasible ratios data exists
                if 'infeasible_ratios' not in data or not data['infeasible_ratios']:
                    log_warning(f"No infeasible ratios data for function {func_name}. Skipping plot.", self.module_name)
                    continue
                
                # Get infeasible data
                infeasible_data = np.array(data['infeasible_ratios'])
                steps = np.arange(len(infeasible_data))
                
                # Create plot
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                
                # Plot infeasible fraction
                ax.plot(steps, infeasible_data, label='Fraction of Infeasible Particles', 
                        color='red', linewidth=2)
                
                ax.set_title(f"Infeasible Particles: {func_name}")
                ax.set_xlabel("PSO Step")
                ax.set_ylabel("Infeasible Fraction")
                ax.set_ylim(0, 1)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add statistics text box
                final_infeasible = infeasible_data[-1] if len(infeasible_data) > 0 else 0
                mean_infeasible = np.mean(infeasible_data) if len(infeasible_data) > 0 else 0
                max_infeasible = np.max(infeasible_data) if len(infeasible_data) > 0 else 0
                min_infeasible = np.min(infeasible_data) if len(infeasible_data) > 0 else 0
                stats_text = f'Final Infeasible: {final_infeasible:.3f}\nMean Infeasible: {mean_infeasible:.3f}\nMax Infeasible: {max_infeasible:.3f}\nMin Infeasible: {min_infeasible:.3f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
                
                plt.tight_layout()
                
                if save_plots and self.checkpoint_base_dir is not None:
                    checkpoint_dir = Path(self.checkpoint_base_dir) / "infeasible_particles"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    timestamped_filename = generate_timestamped_filename(f"infeasible_particles_{func_name}")
                    plt.savefig(checkpoint_dir / timestamped_filename, dpi=300, bbox_inches='tight')
                    log_success(f"Infeasible particles plot saved for {func_name}: {timestamped_filename}", self.module_name)
                if show_plots:
                    plt.show()
                plt.close()
        else:
            log_info("Skipping individual function infeasible particle plots (plot_only_averages=True)", self.module_name)

        # Averaged across all functions
        step_size = ENV_MAX_STEPS
        all_arrs = []
        for func_name in function_names:
            if func_name in metrics_calculator.metric_tracking and 'infeasible_ratio' in metrics_calculator.metric_tracking[func_name]:
                arr = metrics_calculator.metric_tracking[func_name]['infeasible_ratio']
                all_arrs.append(arr)
        if not all_arrs:
            log_warning('No infeasible ratio data for averaging', self.module_name)
            return
        all_arrs = np.concatenate(all_arrs)
        avg_per_step, std_per_step = self._reshape_and_average(all_arrs, step_size)
        steps = np.arange(step_size)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(steps, avg_per_step, label='Fraction of Infeasible Particles', color='red', linewidth=2)
        ax.fill_between(steps, avg_per_step-std_per_step, avg_per_step+std_per_step, color='red', alpha=0.2, label='Std Dev')
        ax.set_title('Infeasible Particles (Averaged Across All Functions)')
        ax.set_xlabel('PSO Step')
        ax.set_ylabel('Infeasible Fraction')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        final_infeasible = avg_per_step[-1] if len(avg_per_step) > 0 else 0
        mean_infeasible = np.nanmean(avg_per_step) if len(avg_per_step) > 0 else 0
        max_infeasible = np.nanmax(avg_per_step) if len(avg_per_step) > 0 else 0
        min_infeasible = np.nanmin(avg_per_step) if len(avg_per_step) > 0 else 0
        stats_text = f'Final Infeasible: {final_infeasible:.3f}\nMean Infeasible: {mean_infeasible:.3f}\nMax Infeasible: {max_infeasible:.3f}\nMin Infeasible: {min_infeasible:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.tight_layout()
        if save_plots and self.checkpoint_base_dir is not None:
            checkpoint_dir = Path(self.checkpoint_base_dir) / "infeasible_particles"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            timestamped_filename = generate_timestamped_filename("infeasible_particles_average")
            plt.savefig(checkpoint_dir / timestamped_filename, dpi=300, bbox_inches='tight')
            log_success(f"Average infeasible particles plot saved: {timestamped_filename}", self.module_name)
        if show_plots:
            plt.show()
        plt.close()

    def plot_all_metrics(self, 
                         metrics_calculator,
                         function_names: Optional[List[str]] = None,
                         save_plots: bool = True,
                         show_plots: bool = False) -> None:
        """
        Generate all available plots for the tracked metrics.
        
        Args:
            metrics_calculator: SwarmMetrics instance with tracking data
            function_names: List of function names to plot. If None, plots all tracked functions.
            save_plots: Whether to save the plots to files
            show_plots: Whether to display the plots
        """
        log_header("Generating all SAPSO metric plots", self.module_name)
        
        # Generate all plot types
        self.plot_parameter_evolution(metrics_calculator, function_names, save_plots, show_plots)
        self.plot_parameter_comparison(metrics_calculator, function_names, save_plots, show_plots)
        self.plot_average_parameters(metrics_calculator, function_names, save_plots, show_plots)
        self.plot_stability_condition(metrics_calculator, function_names, save_plots, show_plots)
        self.plot_infeasible_particles(metrics_calculator, function_names, save_plots, show_plots)
        self.plot_average_velocity(metrics_calculator, function_names, save_plots, show_plots)
        self.plot_swarm_diversity(metrics_calculator, function_names, save_plots, show_plots)
        
        log_success("All SAPSO metric plots generated", self.module_name) 