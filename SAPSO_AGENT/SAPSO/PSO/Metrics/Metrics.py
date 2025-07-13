# File: PSO-ToyBox/SAPSO_AGENT/PSO/Metrics/Metrics.py
# Enhanced to include comprehensive plotting functionality similar to baseline PSO
# Updated to align metric calculations more closely with definitions in
# mathematics-12-03481.pdf (Section 4.1) and accept necessary parameters.
# Added NaN check for input positions.

import numpy as np
import matplotlib.pyplot as plt
import traceback  # For logging exceptions
from pathlib import Path  # To get module name
from typing import Dict, List, Tuple, Optional, Union
import time
from datetime import datetime

from SAPSO_AGENT.Logs.logger import *

# --- Module Name for Logging ---
module_name = Path(__file__).stem  # Gets 'Metrics'


def generate_timestamped_filename(base_name: str, extension: str = "png") -> str:
    """
    Generate a filename with timestamp and base name.
    
    Args:
        base_name: The base name for the file
        extension: File extension (default: "png")
    
    Returns:
        str: Timestamped filename in format "YYYYMMDD_HHMMSS_base_name.extension"
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{base_name}.{extension}"


class SwarmMetrics:
    """
    Enhanced swarm metrics calculator with comprehensive plotting capabilities.
    Calculates swarm metrics using vectorized NumPy operations, aligned with
    definitions in mathematics-12-03481.pdf where possible.
    """

    def __init__(self, checkpoint_base_dir: Optional[str] = None):
        """
        Initializes the enhanced metrics calculator with plotting capabilities.
        
        Args:
            checkpoint_base_dir: Base directory for saving plots and results
        """
        self.checkpoint_base_dir = checkpoint_base_dir
        self.metric_tracking = {}  # Store metric data for plotting
        self.parameter_tracking = {}  # Store parameter data for plotting
        
    def _check_poli_stability(self, omega: float, c1: float, c2: float) -> bool:
        """
        Checks if the given control parameters satisfy Poli's stability condition (Eq. 4).
        """
        if not (-1.0 <= omega <= 1.0):
            return False
        denominator = 7.0 - 5.0 * omega
        if np.isclose(denominator, 0):
            return False
        stability_boundary = 24.0 * (1.0 - omega ** 2) / denominator
        return (c1 + c2) < stability_boundary

    def compute(self, 
                positions: np.ndarray,
                previous_positions: np.ndarray,
                velocities: np.ndarray,
                bounds: tuple,
                omega: float,
                c1: float,
                c2: float,
                step: int = 0,
                function_name: Optional[str] = None,
                run_id: int = 0) -> dict:
        """
        Computes various swarm metrics from the provided state arrays and parameters.
        Now includes data tracking for plotting.
        
        Args:
            positions: Current particle positions
            previous_positions: Previous particle positions
            velocities: Current particle velocities
            bounds: Problem bounds (lower, upper)
            omega: Inertia weight
            c1: Cognitive parameter
            c2: Social parameter
            step: Current optimization step
            function_name: Name of the function being optimized
            run_id: ID of the current run
        """
        # --- Input Validation ---
        if positions.shape[0] == 0 or velocities.shape[
            0] == 0 or previous_positions is None or positions.shape != previous_positions.shape:
            log_warning("Compute called with empty or mismatched arrays.", module_name)
            # Return default dictionary structure with NaN values
            return {
                'avg_step_size': np.nan,
                'avg_current_velocity_magnitude': np.nan,
                'swarm_diversity': np.nan,
                'infeasible_ratio': np.nan,
                'stability_ratio': np.nan
            }

        # Check for NaNs in input positions, which would corrupt diversity calculation
        if np.isnan(positions).any():
            log_warning("NaN values detected in input 'positions' array. Diversity will be NaN.", module_name)
            # Proceed with other calculations, but diversity will likely be NaN

        num_particles = positions.shape[0]
        metrics = {}

        # 1. Average Step Size (Paper's "Average particle velocity" - Eq. 29)
        try:
            step_sizes = np.linalg.norm(positions - previous_positions, axis=1)
            metrics['avg_step_size'] = np.mean(step_sizes)
        except Exception as e:
            log_warning(f"Error calculating avg_step_size: {e}", module_name)
            metrics['avg_step_size'] = np.nan

        # 2. Average Current Velocity Magnitude (For comparison/debugging)
        try:
            velocity_magnitudes = np.linalg.norm(velocities, axis=1)
            metrics['avg_current_velocity_magnitude'] = np.mean(velocity_magnitudes)
        except Exception as e:
            log_warning(f"Error calculating avg_current_velocity_magnitude: {e}", module_name)
            metrics['avg_current_velocity_magnitude'] = np.nan

        # 3. Stability Ratio (Poli's Condition - Eq. 4)
        try:
            is_stable = self._check_poli_stability(omega, c1, c2)
            metrics['stability_ratio'] = 1.0 if is_stable else 0.0
        except Exception as e:
            log_warning(f"Error calculating stability_ratio: {e}", module_name)
            metrics['stability_ratio'] = np.nan

        # 4. Swarm Diversity (Matches Paper Definition - Eq. 27)
        metrics['swarm_diversity'] = np.nan  # Default to NaN
        if num_particles > 1:
            try:
                # Check again for NaNs specifically before diversity calc
                if not np.isnan(positions).any():
                    centroid = np.mean(positions, axis=0)
                    if not np.isnan(centroid).any():  # Ensure centroid is valid
                        distances = np.linalg.norm(positions - centroid, axis=1)
                        metrics['swarm_diversity'] = np.mean(distances)
                    else:
                        log_warning("Centroid calculation resulted in NaN. Diversity set to NaN.", module_name)
                # else: NaN already logged above
            except Exception as e:
                log_warning(f"Error calculating swarm_diversity: {e}", module_name)
                # metrics['swarm_diversity'] remains NaN
        elif num_particles == 1:
            metrics['swarm_diversity'] = 0.0  # Defined as 0 for single particle

        # 5. Infeasible Ratio (Matches Paper Definition)
        try:
            lower_bound, upper_bound = bounds
            is_out_of_bounds = np.any((positions < lower_bound) | (positions > upper_bound), axis=1)
            infeasible_count = np.sum(is_out_of_bounds)
            metrics['infeasible_ratio'] = infeasible_count / num_particles
        except Exception as e:
            log_warning(f"Error calculating infeasible_ratio: {e}", module_name)
            metrics['infeasible_ratio'] = np.nan

        # Track data for plotting if function_name is provided
        if function_name is not None:
            self._track_metrics(metrics, omega, c1, c2, step, function_name, run_id)

        # Final check if any metric is NaN before returning
        if any(np.isnan(v) for v in metrics.values()):
            log_debug(f"Metrics computed with NaNs: {metrics}", module_name)
        else:
            log_debug(f"Computed metrics: {metrics}", module_name)

        return metrics

    def _track_metrics(self, metrics: dict, omega: float, c1: float, c2: float, 
                      step: int, function_name: str, run_id: int):
        """
        Track metrics and parameters for plotting.
        
        Args:
            metrics: Computed metrics dictionary
            omega: Inertia weight
            c1: Cognitive parameter
            c2: Social parameter
            step: Current optimization step
            function_name: Name of the function being optimized
            run_id: ID of the current run
        """
        if function_name not in self.metric_tracking:
            self.metric_tracking[function_name] = {
                'avg_step_size': [],
                'avg_current_velocity_magnitude': [],
                'swarm_diversity': [],
                'infeasible_ratio': [],
                'stability_ratio': [],
                'runs': {}
            }
        
        if function_name not in self.parameter_tracking:
            self.parameter_tracking[function_name] = {
                'omega': [],
                'c1': [],
                'c2': [],
                'runs': {}
            }
        
        # Track metrics
        for metric_name, value in metrics.items():
            if metric_name in self.metric_tracking[function_name]:
                self.metric_tracking[function_name][metric_name].append(value)
        
        # Track parameters
        self.parameter_tracking[function_name]['omega'].append(omega)
        self.parameter_tracking[function_name]['c1'].append(c1)
        self.parameter_tracking[function_name]['c2'].append(c2)
        
        # Track per-run data
        if run_id not in self.metric_tracking[function_name]['runs']:
            self.metric_tracking[function_name]['runs'][run_id] = {
                'avg_step_size': [],
                'avg_current_velocity_magnitude': [],
                'swarm_diversity': [],
                'infeasible_ratio': [],
                'stability_ratio': []
            }
        
        if run_id not in self.parameter_tracking[function_name]['runs']:
            self.parameter_tracking[function_name]['runs'][run_id] = {
                'omega': [],
                'c1': [],
                'c2': []
            }
        
        # Add current step data to run tracking
        for metric_name, value in metrics.items():
            if metric_name in self.metric_tracking[function_name]['runs'][run_id]:
                self.metric_tracking[function_name]['runs'][run_id][metric_name].append(value)
        
        self.parameter_tracking[function_name]['runs'][run_id]['omega'].append(omega)
        self.parameter_tracking[function_name]['runs'][run_id]['c1'].append(c1)
        self.parameter_tracking[function_name]['runs'][run_id]['c2'].append(c2)

    def plot_stability_condition(self, function_names: Optional[List[str]] = None,
                                save_plots: bool = True, show_plots: bool = False) -> None:
        """
        Plot c1+c2 and the stability RHS for each function and averaged across all functions.
        Also plot the fraction of particles that are stable based on the stability condition.
        
        Args:
            function_names: List of function names to plot. If None, plots all tracked functions.
            save_plots: Whether to save the plots to files
            show_plots: Whether to display the plots
        """
        if not self.parameter_tracking:
            log_warning("No parameter tracking data available. Run PSO evaluation first.", module_name)
            return

        if function_names is None:
            function_names = list(self.parameter_tracking.keys())

        # Per-function plots
        for func_name in function_names:
            if func_name not in self.parameter_tracking:
                continue
                
            data = self.parameter_tracking[func_name]
            omega = np.array(data['omega'])
            c1 = np.array(data['c1'])
            c2 = np.array(data['c2'])
            
            # Calculate stability condition
            lhs = c1 + c2
            rhs = 24 * (1 - omega**2) / (7 - 5*omega)
            
            # Calculate stability fraction
            stability_fractions = []
            for i in range(len(omega)):
                if (-1.0 <= omega[i] <= 1.0):
                    denominator = 7.0 - 5.0 * omega[i]
                    if not np.isclose(denominator, 0):
                        stability_boundary = 24.0 * (1.0 - omega[i]**2) / denominator
                        is_stable = (c1[i] + c2[i]) < stability_boundary
                        stability_fractions.append(1.0 if is_stable else 0.0)
                    else:
                        stability_fractions.append(0.0)
                else:
                    stability_fractions.append(0.0)
            
            stability_fractions = np.array(stability_fractions)
            steps = np.arange(len(omega))

            # Create subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot 1: Stability condition
            ax1.plot(steps, lhs, label=r"$c_1 + c_2$", color='blue', linewidth=2)
            ax1.plot(steps, rhs, label=r"$\frac{24(1-w^2)}{7-5w}$", color='orange', linewidth=2)
            ax1.set_title(f"Stability Condition: {func_name}")
            ax1.set_xlabel("Optimization Step")
            ax1.set_ylabel("Value")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Add stability region shading
            ax1.fill_between(steps, 0, rhs, alpha=0.2, color='green', label='Stable Region')
            
            # Plot 2: Stability fraction
            ax2.plot(steps, stability_fractions, label='Fraction of Stable Particles', color='red', linewidth=2)
            ax2.set_title(f"Fraction of Stable Particles: {func_name}")
            ax2.set_xlabel("Optimization Step")
            ax2.set_ylabel("Stability Fraction")
            ax2.set_ylim(0, 1)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add statistics text box
            final_stability = stability_fractions[-1] if len(stability_fractions) > 0 else 0
            mean_stability = np.mean(stability_fractions) if len(stability_fractions) > 0 else 0
            stats_text = f'Final Stability: {final_stability:.3f}\nMean Stability: {mean_stability:.3f}'
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            if save_plots and self.checkpoint_base_dir is not None:
                checkpoint_dir = Path(self.checkpoint_base_dir) / "stability_condition"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                timestamped_filename = generate_timestamped_filename(f"stability_condition_{func_name}")
                plt.savefig(checkpoint_dir / timestamped_filename, dpi=300, bbox_inches='tight')
                log_success(f"Stability condition plot saved for {func_name}: {timestamped_filename}", module_name)
            if show_plots:
                plt.show()
            plt.close()

    def plot_infeasible_particles(self, function_names: Optional[List[str]] = None,
                                 save_plots: bool = True, show_plots: bool = False) -> None:
        """
        Plot the fraction of infeasible particles over optimization steps for each function.
        
        Args:
            function_names: List of function names to plot. If None, plots all tracked functions.
            save_plots: Whether to save the plots to files
            show_plots: Whether to display the plots
        """
        if not self.metric_tracking:
            log_warning("No metric tracking data available. Run PSO evaluation first.", module_name)
            return

        if function_names is None:
            function_names = list(self.metric_tracking.keys())

        log_info(f"Creating infeasible particles plots for {len(function_names)} functions", module_name)

        # Per-function plots
        for func_name in function_names:
            if func_name not in self.metric_tracking:
                log_warning(f"No metric data for function {func_name}", module_name)
                continue

            data = self.metric_tracking[func_name]
            
            if 'infeasible_ratio' not in data or not data['infeasible_ratio']:
                log_warning(f"No infeasible ratio data for function {func_name}. Skipping plot.", module_name)
                continue
            
            # Calculate average infeasible ratio across all runs
            infeasible_data = np.array(data['infeasible_ratio'])
            avg_infeasible_fraction = np.mean(infeasible_data)
            std_infeasible_fraction = np.std(infeasible_data)
            
            # Create plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            steps = np.arange(len(infeasible_data))
            
            # Plot infeasible fraction
            ax.plot(steps, infeasible_data, label='Fraction of Infeasible Particles', 
                    color='red', linewidth=2)
            
            ax.set_title(f"Infeasible Particles: {func_name}")
            ax.set_xlabel("Optimization Step")
            ax.set_ylabel("Infeasible Fraction")
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics text box
            final_infeasible = infeasible_data[-1] if len(infeasible_data) > 0 else 0
            mean_infeasible = np.mean(infeasible_data) if len(infeasible_data) > 0 else 0
            max_infeasible = np.max(infeasible_data) if len(infeasible_data) > 0 else 0
            stats_text = f'Final Infeasible: {final_infeasible:.3f}\nMean Infeasible: {mean_infeasible:.3f}\nMax Infeasible: {max_infeasible:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            if save_plots and self.checkpoint_base_dir is not None:
                checkpoint_dir = Path(self.checkpoint_base_dir) / "infeasible_particles"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                timestamped_filename = generate_timestamped_filename(f"infeasible_particles_{func_name}")
                plt.savefig(checkpoint_dir / timestamped_filename, dpi=300, bbox_inches='tight')
                log_success(f"Infeasible particles plot saved for {func_name}: {timestamped_filename}", module_name)
            if show_plots:
                plt.show()
            plt.close()

    def plot_average_velocity(self, function_names: Optional[List[str]] = None,
                             save_plots: bool = True, show_plots: bool = False) -> None:
        """
        Plot the average velocity magnitude of particles over optimization steps for each function.
        Uses logarithmic scale for better visualization of velocity changes.
        
        Args:
            function_names: List of function names to plot. If None, plots all tracked functions.
            save_plots: Whether to save the plots to files
            show_plots: Whether to display the plots
        """
        if not self.metric_tracking:
            log_warning("No metric tracking data available. Run PSO evaluation first.", module_name)
            return

        if function_names is None:
            function_names = list(self.metric_tracking.keys())

        log_info(f"Creating average velocity plots for {len(function_names)} functions", module_name)

        # Per-function plots
        for func_name in function_names:
            if func_name not in self.metric_tracking:
                log_warning(f"No metric data for function {func_name}", module_name)
                continue

            data = self.metric_tracking[func_name]
            
            if 'avg_step_size' not in data or not data['avg_step_size']:
                log_warning(f"No velocity data for function {func_name}. Skipping plot.", module_name)
                continue
            
            # Calculate average velocity across all runs
            velocity_data = np.array(data['avg_step_size'])
            avg_velocity = np.mean(velocity_data)
            std_velocity = np.std(velocity_data)
            
            # Create plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            steps = np.arange(len(velocity_data))
            
            # Plot average velocity with error bands on log scale
            ax.semilogy(steps, velocity_data, label='Average Velocity Magnitude', 
                       color='blue', linewidth=2)
            
            ax.set_title(f"Average Particle Velocity: {func_name}")
            ax.set_xlabel("Optimization Step")
            ax.set_ylabel("Average Velocity Magnitude (log scale)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics text box
            final_velocity = velocity_data[-1] if len(velocity_data) > 0 else 0
            mean_velocity = np.mean(velocity_data) if len(velocity_data) > 0 else 0
            max_velocity = np.max(velocity_data) if len(velocity_data) > 0 else 0
            min_velocity = np.min(velocity_data) if len(velocity_data) > 0 else 0
            stats_text = f'Final Velocity: {final_velocity:.2e}\nMean Velocity: {mean_velocity:.2e}\nMax Velocity: {max_velocity:.2e}\nMin Velocity: {min_velocity:.2e}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            
            if save_plots and self.checkpoint_base_dir is not None:
                checkpoint_dir = Path(self.checkpoint_base_dir) / "average_velocity"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                timestamped_filename = generate_timestamped_filename(f"average_velocity_{func_name}")
                plt.savefig(checkpoint_dir / timestamped_filename, dpi=300, bbox_inches='tight')
                log_success(f"Average velocity plot saved for {func_name}: {timestamped_filename}", module_name)
            if show_plots:
                plt.show()
            plt.close()

    def plot_swarm_diversity(self, function_names: Optional[List[str]] = None,
                            save_plots: bool = True, show_plots: bool = False,
                            clip_std_dev: bool = False) -> None:
        """
        Plot the swarm diversity over optimization steps for each function.
        
        Args:
            function_names: List of function names to plot. If None, plots all tracked functions.
            save_plots: Whether to save the plots to files
            show_plots: Whether to display the plots
            clip_std_dev: Whether to clip the standard deviation bands to avoid extreme values
        """
        if not self.metric_tracking:
            log_warning("No metric tracking data available. Run PSO evaluation first.", module_name)
            return

        if function_names is None:
            function_names = list(self.metric_tracking.keys())

        log_info(f"Creating swarm diversity plots for {len(function_names)} functions", module_name)

        # Per-function plots
        for func_name in function_names:
            if func_name not in self.metric_tracking:
                log_warning(f"No metric data for function {func_name}", module_name)
                continue

            data = self.metric_tracking[func_name]
            
            if 'swarm_diversity' not in data or not data['swarm_diversity']:
                log_warning(f"No diversity data for function {func_name}. Skipping plot.", module_name)
                continue
            
            try:
                diversity_data = np.array(data['swarm_diversity'])
                avg_diversity = np.mean(diversity_data)
                std_diversity = np.std(diversity_data)
                steps = np.arange(len(diversity_data))

                plt.figure(figsize=(10, 6))
                plt.semilogy(steps, diversity_data, label='Average Swarm Diversity', color='green', linewidth=2)
                
                if not clip_std_dev:
                    plt.fill_between(steps, 
                                   np.maximum(avg_diversity - std_diversity, 1e-10),  # Avoid negative values for log scale
                                   avg_diversity + std_diversity,
                                   alpha=0.3, color='green', label='±1 Std Dev')
                
                plt.xlabel('Optimization Step')
                plt.ylabel('Swarm Diversity (log scale)')
                title_suffix = " (Clipped)" if clip_std_dev else ""
                plt.title(f'Swarm Diversity over Steps ({func_name}){title_suffix}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Add statistics text box
                final_diversity = diversity_data[-1] if len(diversity_data) > 0 else 0
                mean_diversity = np.mean(diversity_data) if len(diversity_data) > 0 else 0
                max_diversity = np.max(diversity_data) if len(diversity_data) > 0 else 0
                min_diversity = np.min(diversity_data) if len(diversity_data) > 0 else 0
                stats_text = f'Final Diversity: {final_diversity:.4f}\nMean Diversity: {mean_diversity:.4f}\nMax Diversity: {max_diversity:.4f}\nMin Diversity: {min_diversity:.4f}'
                plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
                
                plt.tight_layout()
                
                if save_plots and self.checkpoint_base_dir is not None:
                    checkpoint_dir = Path(self.checkpoint_base_dir) / "swarm_diversity"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    timestamped_filename = generate_timestamped_filename(f"swarm_diversity_{func_name}")
                    plt.savefig(checkpoint_dir / timestamped_filename, dpi=300, bbox_inches='tight')
                    log_success(f"Swarm diversity plot saved for {func_name}: {timestamped_filename}", module_name)
                elif save_plots:
                    timestamped_filename = generate_timestamped_filename(f"metrics_swarm_diversity_{func_name}")
                    plt.savefig(timestamped_filename, dpi=300, bbox_inches='tight')
                    log_success(f"Swarm diversity plot saved for {func_name}: {timestamped_filename}", module_name)
                
                if show_plots:
                    plt.show()
                plt.close()
                
            except Exception as e:
                log_error(f"Error creating swarm diversity plot for {func_name}: {e}", module_name)
                plt.close()

    def plot_parameter_evolution(self, function_names: Optional[List[str]] = None,
                                save_plots: bool = True, show_plots: bool = False) -> None:
        """
        Plot the average control parameter values (omega, c1, c2) over the optimization steps.
        
        Args:
            function_names: List of function names to plot. If None, plots all tracked functions.
            save_plots: Whether to save the plots to files
            show_plots: Whether to display the plots
        """
        if not self.parameter_tracking:
            log_warning("No parameter tracking data available. Run PSO evaluation first.", module_name)
            return
        
        if function_names is None:
            function_names = list(self.parameter_tracking.keys())
        
        log_info(f"Plotting parameter evolution for {len(function_names)} functions", module_name)
        
        # Create figure with subplots for each function
        num_functions = len(function_names)
        fig, axes = plt.subplots(num_functions, 1, figsize=(12, 4 * num_functions))
        
        # Ensure axes is always a list
        if num_functions == 1:
            axes = [axes]
        
        for i, func_name in enumerate(function_names):
            if func_name not in self.parameter_tracking:
                log_warning(f"No parameter data for function {func_name}", module_name)
                continue
            
            tracking_data = self.parameter_tracking[func_name]
            
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
            
            timestamped_filename = generate_timestamped_filename(f"parameter_evolution_{func_suffix}")
            plot_path = checkpoint_dir / timestamped_filename
            
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            log_success(f"Parameter evolution plot saved to {plot_path}", module_name)
        
        # Show plot if requested
        if show_plots:
            plt.show()
        
        plt.close()

    def plot_all_metrics(self, function_names: Optional[List[str]] = None,
                         save_plots: bool = True, show_plots: bool = False) -> None:
        """
        Generate all available plots for the tracked metrics.
        
        Args:
            function_names: List of function names to plot. If None, plots all tracked functions.
            save_plots: Whether to save the plots to files
            show_plots: Whether to display the plots
        """
        log_header("Generating all metric plots", module_name)
        
        # Generate all plot types
        self.plot_stability_condition(function_names, save_plots, show_plots)
        self.plot_infeasible_particles(function_names, save_plots, show_plots)
        self.plot_average_velocity(function_names, save_plots, show_plots)
        self.plot_swarm_diversity(function_names, save_plots, show_plots)
        self.plot_parameter_evolution(function_names, save_plots, show_plots)
        
        log_success("All metric plots generated", module_name)

    def clear_tracking_data(self):
        """Clear all tracked metric and parameter data."""
        self.metric_tracking.clear()
        self.parameter_tracking.clear()
        log_info("Cleared all tracking data", module_name)

    def get_tracking_summary(self) -> dict:
        """
        Get a summary of tracked data.
        
        Returns:
            Dictionary containing summary information about tracked data
        """
        summary = {
            'functions_tracked': list(self.metric_tracking.keys()),
            'parameter_functions': list(self.parameter_tracking.keys()),
            'total_metric_points': sum(len(data.get('swarm_diversity', [])) for data in self.metric_tracking.values()),
            'total_parameter_points': sum(len(data.get('omega', [])) for data in self.parameter_tracking.values())
        }
        return summary 