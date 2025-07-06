"""
Baseline PSO Implementation without Reinforcement Learning

This module provides a baseline implementation that runs PSO directly without any RL agent.
It imports the training and test functions from SAPSO_AGENT and runs PSO with different parameter strategies.
"""

import random
import time
import numpy as np
import matplotlib.pyplot as plt
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

# Import the training and test functions from SAPSO_AGENT
from SAPSO_AGENT.Benchmark.train import train_agent as sapso_train_agent
from SAPSO_AGENT.Benchmark.test import test_agent as sapso_test_agent

# Import PSO components
from SAPSO_AGENT.SAPSO.PSO.PSO import PSOVectorized
from SAPSO_AGENT.SAPSO.PSO.Cognitive.GBest import GlobalBestStrategy
from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.Training.Loader import objective_function_classes
from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.Testing.Loader import test_objective_function_classes

# Import configuration and logging
from SAPSO_AGENT.CONFIG import *
from SAPSO_AGENT.Logs.logger import *

# Import graphics for plotting
from SAPSO_AGENT.SAPSO.Graphics.graphing import (
    plot_evaluation_parameters,
    plot_stable_particles,
    plot_infeasible_particles,
    plot_average_velocity,
    plot_swarm_diversity,
    plot_gbest_convergence,
    plot_final_gbest_per_function,
    plot_gbest_convergence_per_function
)

# Import parameter strategies
from Baseline.parameter_strategies import ParameterStrategy, create_strategy, list_available_strategies


class BaselinePSO:
    """
    Baseline PSO implementation that runs PSO without any RL agent.
    
    This class provides methods to run PSO with different parameter strategies on the same
    objective functions used in the SAPSO training and testing.
    """
    
    def __init__(self, 
                 env_dim: int = ENV_DIM,
                 env_particles: int = ENV_PARTICLES,
                 env_max_steps: int = ENV_MAX_STEPS,
                 v_clamp_ratio: float = 0.2,
                 use_velocity_clamping: bool = USE_VELOCITY_CLAMPING,
                 convergence_patience: int = 50,
                 convergence_threshold_gbest: float = 1e-8,
                 convergence_threshold_pbest_std: float = 1e-6,
                 checkpoint_base_dir: Optional[str] = CHECKPOINT_BASE_DIR,
                 parameter_strategy: Union[str, ParameterStrategy] = 'fixed',
                 strategy_kwargs: Optional[Dict] = None):
        """
        Initialize the Baseline PSO system.
        
        Args:
            env_dim: Problem dimension
            env_particles: Number of particles in the swarm
            env_max_steps: Maximum PSO steps per run
            v_clamp_ratio: Velocity clamping ratio
            use_velocity_clamping: Whether to use velocity clamping
            convergence_patience: Convergence patience parameter
            convergence_threshold_gbest: GBest convergence threshold
            convergence_threshold_pbest_std: PBest standard deviation convergence threshold
            checkpoint_base_dir: Base directory for saving results
            parameter_strategy: Parameter strategy to use (string name or ParameterStrategy instance)
            strategy_kwargs: Additional keyword arguments for the parameter strategy
        """
        self.env_dim = env_dim
        self.env_particles = env_particles
        self.env_max_steps = env_max_steps
        self.v_clamp_ratio = v_clamp_ratio
        self.use_velocity_clamping = use_velocity_clamping
        self.convergence_patience = convergence_patience
        self.convergence_threshold_gbest = convergence_threshold_gbest
        self.convergence_threshold_pbest_std = convergence_threshold_pbest_std
        self.checkpoint_base_dir = checkpoint_base_dir
        
        # Initialize parameter strategy
        self.strategy_kwargs = strategy_kwargs or {}
        if isinstance(parameter_strategy, str):
            self.parameter_strategy = create_strategy(parameter_strategy, **self.strategy_kwargs)
        elif isinstance(parameter_strategy, ParameterStrategy):
            self.parameter_strategy = parameter_strategy
        else:
            raise ValueError("parameter_strategy must be a string or ParameterStrategy instance")
        
        # Results storage
        self.training_results = {}
        self.testing_results = {}
        self.diversity_results = {}
        
        self.module_name = Path(__file__).stem
        
        log_info(f"Initialized Baseline PSO with strategy: {self.parameter_strategy}", self.module_name)
        
    def run_pso_on_function(self, 
                           obj_func_class, 
                           num_runs: int = 1,
                           function_name: Optional[str] = None,
                           allow_convergence_stopping: bool = False) -> Tuple[List[Tuple[float, int]], List[List[float]]]:
        """
        Run PSO on a single objective function multiple times.
        
        Args:
            obj_func_class: The objective function class to use
            num_runs: Number of independent runs
            function_name: Name of the function for logging
            allow_convergence_stopping: Whether to stop the PSO when the GBest converges
            
        Returns:
            Tuple containing list of tuples (final_gbest_value, steps_taken) for each run and list of lists of diversity values
        """
        if function_name is None:
            function_name = obj_func_class.__name__
            
        log_info(f"Running baseline PSO on {function_name} for {num_runs} runs", self.module_name)
        log_info(f"Parameter strategy: {self.parameter_strategy}", self.module_name)
        
        results = []
        all_diversities = []
        # Track parameter values over all runs for this function
        all_omega_values = []
        all_c1_values = []
        all_c2_values = []
        # Track infeasible ratios over all runs for this function
        all_infeasible_ratios = []
        # Track average velocities over all runs for this function
        all_avg_velocities = []
        
        for run in range(num_runs):
            try:
                # Create objective function instance
                current_dim = self.env_dim
                if function_name == "GiuntaFunction":
                    current_dim = 2
                    log_info(f"  Adjusting dimension to {current_dim} for {function_name}", self.module_name)
                
                obj_func_instance = obj_func_class(dim=current_dim)
                
                # Initialize PSO
                pso = PSOVectorized(
                    objective_function=obj_func_instance,
                    num_particles=self.env_particles,
                    strategy=None,  # Don't use strategy for baseline
                    v_clamp_ratio=self.v_clamp_ratio,
                    use_velocity_clamping=self.use_velocity_clamping,
                    convergence_patience=self.convergence_patience,
                    convergence_threshold_gbest=self.convergence_threshold_gbest,
                    convergence_threshold_pbest_std=self.convergence_threshold_pbest_std
                )
                
                # Reset parameter strategy for new run
                self.parameter_strategy.reset()
                
                # Track parameters, infeasible ratios, and velocities for this run
                run_omega_values = []
                run_c1_values = []
                run_c2_values = []
                run_infeasible_ratios = []
                run_avg_velocities = []
                run_diversities = []
                
                # Run PSO with dynamic parameters
                steps_taken = 0
                converged = False
                
                for step in range(self.env_max_steps):
                    # Get parameters from strategy
                    omega, c1, c2 = self.parameter_strategy.get_parameters(
                        step=step, 
                        max_steps=self.env_max_steps,
                        gbest_value=pso.gbest_value
                    )
                    
                    # Track parameter values
                    run_omega_values.append(omega)
                    run_c1_values.append(c1)
                    run_c2_values.append(c2)
                    
                    # Run PSO step with current parameters
                    metrics, gbest_value, converged = pso.optimize_step(omega, c1, c2)
                    
                    # Calculate infeasible ratio for this step
                    infeasible_ratio = self._calculate_infeasible_ratio(pso, obj_func_instance)
                    run_infeasible_ratios.append(infeasible_ratio)
                    
                    # Calculate average velocity magnitude for this step
                    avg_velocity = self._calculate_average_velocity(pso)
                    run_avg_velocities.append(avg_velocity)

                    # Calculate swarm diversity for this step
                    diversity = self._calculate_swarm_diversity(pso)
                    run_diversities.append(diversity)
                    
                    steps_taken += 1
                    
                    if converged and allow_convergence_stopping:
                        log_debug(f"  Run {run + 1}: Converged at step {steps_taken}", self.module_name)
                        break
                
                # Pad shorter runs with the last parameter values to maintain consistent length
                while len(run_omega_values) < self.env_max_steps:
                    run_omega_values.append(run_omega_values[-1])
                    run_c1_values.append(run_c1_values[-1])
                    run_c2_values.append(run_c2_values[-1])
                    run_infeasible_ratios.append(run_infeasible_ratios[-1])
                    run_avg_velocities.append(run_avg_velocities[-1])
                
                # Add this run's data to the overall tracking
                all_omega_values.append(run_omega_values)
                all_c1_values.append(run_c1_values)
                all_c2_values.append(run_c2_values)
                all_infeasible_ratios.append(run_infeasible_ratios)
                all_avg_velocities.append(run_avg_velocities)
                all_diversities.append(run_diversities)
                
                final_gbest = pso.gbest_value
                results.append((final_gbest, steps_taken))
                
                log_debug(f"  Run {run + 1}: Final GBest = {final_gbest:.6e}, Steps = {steps_taken}", self.module_name)
                
            except Exception as e:
                log_error(f"Error in run {run + 1} for {function_name}: {e}", self.module_name)
                log_error(traceback.format_exc(), self.module_name)
                results.append((float('inf'), self.env_max_steps))  # Failed run
        
        # Store parameter tracking data for this function
        if not hasattr(self, 'parameter_tracking'):
            self.parameter_tracking = {}
        
        self.parameter_tracking[function_name] = {
            'omega': all_omega_values,
            'c1': all_c1_values,
            'c2': all_c2_values,
            'infeasible_ratios': all_infeasible_ratios,
            'avg_velocities': all_avg_velocities,
            'num_runs': num_runs
        }
        
        return results, all_diversities
    
    def _calculate_infeasible_ratio(self, pso, obj_func_instance):
        """
        Calculate the fraction of particles that are infeasible (out of bounds).
        
        Args:
            pso: The PSO instance
            obj_func_instance: The objective function instance
            
        Returns:
            float: Fraction of particles that are infeasible (0.0 to 1.0)
        """
        try:
            # Get current particle positions
            positions = pso.positions
            
            # Get bounds from objective function
            lower_bound, upper_bound = obj_func_instance.bounds
            
            # Check if any dimension is out of bounds for each particle
            is_out_of_bounds = np.any((positions < lower_bound) | (positions > upper_bound), axis=1)
            
            # Calculate infeasible ratio
            infeasible_count = np.sum(is_out_of_bounds)
            infeasible_ratio = infeasible_count / pso.num_particles
            
            return infeasible_ratio
            
        except Exception as e:
            log_warning(f"Error calculating infeasible ratio: {e}", self.module_name)
            return 0.0  # Default to 0 if calculation fails
    
    def _calculate_average_velocity(self, pso):
        """
        Calculate the average movement (distance) of all particles between steps,
        as Δ(t+1) = (1/ns) * sum_i ||x_i(t+1) - x_i(t)||.
        
        Args:
            pso: The PSO instance
            
        Returns:
            float: Average movement magnitude
        """
        try:
            # Get current and previous particle positions
            current_positions = pso.positions
            previous_positions = pso.previous_positions

            # Calculate movement magnitudes for each particle
            movement_magnitudes = np.linalg.norm(current_positions - previous_positions, axis=1)
            
            # Calculate average movement magnitude
            avg_movement = np.mean(movement_magnitudes)
            
            return avg_movement
            
        except Exception as e:
            log_warning(f"Error calculating average movement: {e}", self.module_name)
            return 0.0  # Default to 0 if calculation fails
    
    def _calculate_swarm_diversity(self, pso):
        """
        Calculate the average swarm diversity as per the provided formula.
        Args:
            pso: The PSO instance
        Returns:
            float: Average swarm diversity
        """
        try:
            positions = pso.positions  # shape: (n_s, n_x)
            swarm_center = np.mean(positions, axis=0)  # shape: (n_x,)
            # Compute Euclidean distance of each particle to the swarm center
            distances = np.linalg.norm(positions - swarm_center, axis=1)
            diversity = np.mean(distances)
            return diversity
        except Exception as e:
            log_warning(f"Error calculating swarm diversity: {e}", self.module_name)
            return 0.0
    
    def run_training_functions(self, 
                              episodes_per_function: int = EPISODES_PER_FUNCTION,
                              max_functions: Optional[int] = None) -> Dict[str, List[Tuple[float, int]]]:
        """
        Run baseline PSO on training functions.
        
        Args:
            episodes_per_function: Number of runs per function
            max_functions: Maximum number of functions to test (None for all)
            
        Returns:
            Dictionary mapping function names to results
        """
        log_header("Starting baseline PSO training functions evaluation", self.module_name)
        
        functions_to_test = objective_function_classes
        if max_functions is not None:
            functions_to_test = functions_to_test[:max_functions]
        
        self.training_results = {}
        
        for func_index, func_class in enumerate(functions_to_test):
            func_name = func_class.__name__
            log_header(f"===== Baseline PSO on Training Function {func_index + 1}/{len(functions_to_test)}: {func_name} =====", self.module_name)
            
            results, diversities = self.run_pso_on_function(func_class, episodes_per_function, func_name)
            self.training_results[func_name] = results
            self.diversity_results[func_name] = diversities
            
            # Calculate statistics
            finite_gbests = [g for g, s in results if np.isfinite(g)]
            if finite_gbests:
                mean_gbest = np.mean(finite_gbests)
                std_gbest = np.std(finite_gbests)
                log_info(f"  Average Final GBest: {mean_gbest:.6e} +/- {std_gbest:.6e}", self.module_name)
            else:
                log_warning(f"  No finite GBest values recorded for {func_name}", self.module_name)
        
        log_header("Baseline PSO training functions evaluation complete", self.module_name)
        return self.training_results
    
    def run_testing_functions(self, 
                             num_eval_runs: int = NUM_EVAL_RUNS) -> Dict[str, List[Tuple[float, int]]]:
        """
        Run baseline PSO on testing functions.
        
        Args:
            num_eval_runs: Number of evaluation runs per function
            
        Returns:
            Dictionary mapping function names to results
        """
        log_header("Starting baseline PSO testing functions evaluation", self.module_name)
        
        self.testing_results = {}
        
        for func_index, func_class in enumerate(test_objective_function_classes):
            func_name = func_class.__name__
            log_header(f"===== Baseline PSO on Testing Function {func_index + 1}/{len(test_objective_function_classes)}: {func_name} =====", self.module_name)
            
            results, all_diversities = self.run_pso_on_function(func_class, num_eval_runs, func_name)
            self.testing_results[func_name] = results
            self.diversity_results[func_name] = all_diversities
            
            # Calculate statistics
            finite_gbests = [g for g, s in results if np.isfinite(g)]
            if finite_gbests:
                mean_gbest = np.mean(finite_gbests)
                std_gbest = np.std(finite_gbests)
                log_info(f"  Average Final GBest: {mean_gbest:.6e} +/- {std_gbest:.6e}", self.module_name)
            else:
                log_warning(f"  No finite GBest values recorded for {func_name}", self.module_name)
        
        log_header("Baseline PSO testing functions evaluation complete", self.module_name)
        return self.testing_results
    
    def save_results(self, results_type: str = "both"):
        """
        Save results to files.
        
        Args:
            results_type: "training", "testing", or "both"
        """
        if self.checkpoint_base_dir is None:
            log_warning("No checkpoint directory specified. Results not saved.", self.module_name)
            return
        
        checkpoint_dir = Path(self.checkpoint_base_dir) / "baseline_pso_results"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        strategy_name = self.parameter_strategy.__class__.__name__
        
        if results_type in ["training", "both"] and self.training_results:
            training_file = checkpoint_dir / f"baseline_{strategy_name}_training_results_{timestamp}.txt"
            self._save_results_to_file(self.training_results, training_file, "Training")
        
        if results_type in ["testing", "both"] and self.testing_results:
            testing_file = checkpoint_dir / f"baseline_{strategy_name}_testing_results_{timestamp}.txt"
            self._save_results_to_file(self.testing_results, testing_file, "Testing")
    
    def _save_results_to_file(self, results: Dict[str, List[Tuple[float, int]]], 
                             filepath: Path, results_type: str):
        """Helper method to save results to a text file."""
        try:
            with open(filepath, 'w') as f:
                f.write(f"Baseline PSO {results_type} Results\n")
                f.write(f"Parameter Strategy: {self.parameter_strategy}\n")
                f.write(f"Strategy Parameters: {self.strategy_kwargs}\n")
                f.write(f"Environment: dim={self.env_dim}, particles={self.env_particles}, max_steps={self.env_max_steps}\n")
                f.write("=" * 80 + "\n\n")
                
                for func_name, func_results in results.items():
                    f.write(f"Function: {func_name}\n")
                    f.write("-" * 40 + "\n")
                    
                    finite_gbests = [g for g, s in func_results if np.isfinite(g)]
                    if finite_gbests:
                        mean_gbest = np.mean(finite_gbests)
                        std_gbest = np.std(finite_gbests)
                        min_gbest = np.min(finite_gbests)
                        max_gbest = np.max(finite_gbests)
                        
                        f.write(f"  Mean GBest: {mean_gbest:.6e}\n")
                        f.write(f"  Std GBest: {std_gbest:.6e}\n")
                        f.write(f"  Min GBest: {min_gbest:.6e}\n")
                        f.write(f"  Max GBest: {max_gbest:.6e}\n")
                        f.write(f"  Successful runs: {len(finite_gbests)}/{len(func_results)}\n")
                    else:
                        f.write(f"  No successful runs\n")
                    
                    f.write("\n")
            
            log_success(f"Baseline {results_type} results saved to {filepath}", self.module_name)
            
        except Exception as e:
            log_error(f"Error saving {results_type} results: {e}", self.module_name)
    
    def run_full_baseline(self, 
                          episodes_per_function: int = EPISODES_PER_FUNCTION,
                          num_eval_runs: int = NUM_EVAL_RUNS,
                          max_training_functions: Optional[int] = None) -> Dict[str, Dict[str, List[Tuple[float, int]]]]:
        """
        Run the complete baseline evaluation (training + testing functions).
        
        Args:
            episodes_per_function: Number of runs per training function
            num_eval_runs: Number of runs per testing function
            max_training_functions: Maximum number of training functions to test
            
        Returns:
            Dictionary containing both training and testing results
        """
        log_header("Starting complete baseline PSO evaluation", self.module_name)
        start_time = time.time()
        
        # Run training functions
        training_results = self.run_training_functions(episodes_per_function, max_training_functions)
        
        # Run testing functions
        testing_results = self.run_testing_functions(num_eval_runs)
        
        # Save results
        self.save_results("testing")
        
        end_time = time.time()
        log_header(f"Complete baseline PSO evaluation finished in {end_time - start_time:.2f} seconds", self.module_name)
        
        return {
            "training": training_results,
            "testing": testing_results
        }
    
    def plot_parameter_evolution(self, 
                                function_names: Optional[List[str]] = None,
                                save_plots: bool = True,
                                show_plots: bool = False) -> None:
        """
        Plot the average control parameter values (omega, c1, c2) over the optimization steps.
        
        Args:
            function_names: List of function names to plot. If None, plots all tracked functions.
            save_plots: Whether to save the plots to files
            show_plots: Whether to display the plots
        """
        if not hasattr(self, 'parameter_tracking') or not self.parameter_tracking:
            log_warning("No parameter tracking data available. Run PSO evaluation first.", self.module_name)
            return
        
        if function_names is None:
            function_names = list(self.parameter_tracking.keys())
        
        log_info(f"Plotting parameter evolution for {len(function_names)} functions", self.module_name)
        
        # Create figure with subplots for each function
        num_functions = len(function_names)
        fig, axes = plt.subplots(num_functions, 1, figsize=(12, 4 * num_functions))
        
        # Ensure axes is always a list
        if num_functions == 1:
            axes = [axes]
        
        steps = np.arange(self.env_max_steps)
        
        for i, func_name in enumerate(function_names):
            if func_name not in self.parameter_tracking:
                log_warning(f"No parameter data for function {func_name}", self.module_name)
                continue
            
            tracking_data = self.parameter_tracking[func_name]
            
            # Calculate average parameter values across all runs
            omega_avg = np.mean(tracking_data['omega'], axis=0)
            c1_avg = np.mean(tracking_data['c1'], axis=0)
            c2_avg = np.mean(tracking_data['c2'], axis=0)
            
            # Calculate standard deviation for error bands
            omega_std = np.std(tracking_data['omega'], axis=0)
            c1_std = np.std(tracking_data['c1'], axis=0)
            c2_std = np.std(tracking_data['c2'], axis=0)
            
            ax = axes[i]
            
            # Plot average values with error bands
            ax.plot(steps, omega_avg, 'b-', label='ω (Inertia)', linewidth=2)
            ax.fill_between(steps, omega_avg - omega_std, omega_avg + omega_std, 
                          alpha=0.3, color='blue')
            
            ax.plot(steps, c1_avg, 'r-', label='c₁ (Cognitive)', linewidth=2)
            ax.fill_between(steps, c1_avg - c1_std, c1_avg + c1_std, 
                          alpha=0.3, color='red')
            
            ax.plot(steps, c2_avg, 'g-', label='c₂ (Social)', linewidth=2)
            ax.fill_between(steps, c2_avg - c2_std, c2_avg + c2_std, 
                          alpha=0.3, color='green')
            
            ax.set_xlabel('Optimization Step')
            ax.set_ylabel('Parameter Value')
            ax.set_title(f'Parameter Evolution: {func_name}\n'
                        f'Strategy: {self.parameter_strategy.__class__.__name__}, '
                        f'Runs: {tracking_data["num_runs"]}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Set y-axis limits to show the full range
            all_values = np.concatenate([omega_avg, c1_avg, c2_avg])
            y_min, y_max = np.min(all_values), np.max(all_values)
            y_range = y_max - y_min
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plots and self.checkpoint_base_dir is not None:
            checkpoint_dir = Path(self.checkpoint_base_dir) / "parameter_evolution"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = int(time.time())
            strategy_name = self.parameter_strategy.__class__.__name__
            
            # Create filename with function names
            func_suffix = "_".join([name.replace("Function", "") for name in function_names[:3]])
            if len(function_names) > 3:
                func_suffix += f"_and_{len(function_names)-3}_more"
            
            plot_filename = f"baseline_{strategy_name}_parameter_evolution_{func_suffix}_{timestamp}.png"
            plot_path = checkpoint_dir / plot_filename
            
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            log_success(f"Parameter evolution plot saved to {plot_path}", self.module_name)
        
        # Show plot if requested
        if show_plots:
            plt.show()
        
        plt.close()
    
    def plot_parameter_comparison(self, 
                                 function_names: Optional[List[str]] = None,
                                 save_plots: bool = True,
                                 show_plots: bool = False) -> None:
        """
        Create a comparison plot showing parameter evolution across multiple functions.
        
        Args:
            function_names: List of function names to plot. If None, plots all tracked functions.
            save_plots: Whether to save the plots to files
            show_plots: Whether to display the plots
        """
        if not hasattr(self, 'parameter_tracking') or not self.parameter_tracking:
            log_warning("No parameter tracking data available. Run PSO evaluation first.", self.module_name)
            return
        
        if function_names is None:
            function_names = list(self.parameter_tracking.keys())
        
        log_info(f"Creating parameter comparison plot for {len(function_names)} functions", self.module_name)
        
        # Create figure with subplots for each parameter
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        steps = np.arange(self.env_max_steps)
        # Use a simple color palette
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for param_idx, (param_name, param_key) in enumerate([('ω (Inertia)', 'omega'), 
                                                            ('c₁ (Cognitive)', 'c1'), 
                                                            ('c₂ (Social)', 'c2')]):
            ax = axes[param_idx]
            
            for i, func_name in enumerate(function_names):
                if func_name not in self.parameter_tracking:
                    continue
                
                tracking_data = self.parameter_tracking[func_name]
                param_avg = np.mean(tracking_data[param_key], axis=0)
                
                # Shorten function name for legend
                short_name = func_name.replace("Function", "")
                ax.plot(steps, param_avg, color=colors[i], label=short_name, linewidth=1.5)
            
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
            
            timestamp = int(time.time())
            strategy_name = self.parameter_strategy.__class__.__name__
            
            plot_filename = f"baseline_{strategy_name}_parameter_comparison_{timestamp}.png"
            plot_path = checkpoint_dir / plot_filename
            
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            log_success(f"Parameter comparison plot saved to {plot_path}", self.module_name)
        
        # Show plot if requested
        if show_plots:
            plt.show()
        
        plt.close()
    
    def plot_average_parameters(self, 
                               function_names: Optional[List[str]] = None,
                               save_plots: bool = True,
                               show_plots: bool = False) -> None:
        """
        Plot the average control parameter values across all functions in a single graph.
        
        Args:
            function_names: List of function names to include. If None, uses all tracked functions.
            save_plots: Whether to save the plot to file
            show_plots: Whether to display the plot
        """
        if not hasattr(self, 'parameter_tracking') or not self.parameter_tracking:
            log_warning("No parameter tracking data available. Run PSO evaluation first.", self.module_name)
            return
        
        if function_names is None:
            function_names = list(self.parameter_tracking.keys())
        
        log_info(f"Creating average parameter plot across {len(function_names)} functions", self.module_name)
        
        # Create single figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        steps = np.arange(self.env_max_steps)
        
        # Collect all parameter data across functions
        all_omega_data = []
        all_c1_data = []
        all_c2_data = []
        
        for func_name in function_names:
            if func_name not in self.parameter_tracking:
                continue
            
            tracking_data = self.parameter_tracking[func_name]
            
            # Add all runs for this function
            all_omega_data.extend(tracking_data['omega'])
            all_c1_data.extend(tracking_data['c1'])
            all_c2_data.extend(tracking_data['c2'])
        
        if not all_omega_data:
            log_warning("No parameter data available for plotting", self.module_name)
            return
        
        # Calculate overall averages across all functions and runs
        omega_avg = np.mean(all_omega_data, axis=0)
        c1_avg = np.mean(all_c1_data, axis=0)
        c2_avg = np.mean(all_c2_data, axis=0)
        
        # Calculate standard deviation across all functions and runs
        omega_std = np.std(all_omega_data, axis=0)
        c1_std = np.std(all_c1_data, axis=0)
        c2_std = np.std(all_c2_data, axis=0)
        
        # Plot average values with error bands
        ax.plot(steps, omega_avg, 'b-', label='ω (Inertia)', linewidth=3)
        ax.fill_between(steps, omega_avg - omega_std, omega_avg + omega_std, 
                       alpha=0.3, color='blue')
        
        ax.plot(steps, c1_avg, 'r-', label='c₁ (Cognitive)', linewidth=3)
        ax.fill_between(steps, c1_avg - c1_std, c1_avg + c1_std, 
                       alpha=0.3, color='red')
        
        ax.plot(steps, c2_avg, 'g-', label='c₂ (Social)', linewidth=3)
        ax.fill_between(steps, c2_avg - c2_std, c2_avg + c2_std, 
                       alpha=0.3, color='green')
        
        ax.set_xlabel('Optimization Step', fontsize=12)
        ax.set_ylabel('Parameter Value', fontsize=12)
        ax.set_title(f'Average Control Parameters Across All Functions\n'
                    f'Strategy: {self.parameter_strategy.__class__.__name__}, '
                    f'Functions: {len(function_names)}, '
                    f'Total Runs: {len(all_omega_data)}', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits to show the full range
        all_values = np.concatenate([omega_avg, c1_avg, c2_avg])
        y_min, y_max = np.min(all_values), np.max(all_values)
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
        
        # Add statistics text box
        stats_text = f'Final Values:\nω = {omega_avg[-1]:.3f} ± {omega_std[-1]:.3f}\nc₁ = {c1_avg[-1]:.3f} ± {c1_std[-1]:.3f}\nc₂ = {c2_avg[-1]:.3f} ± {c2_std[-1]:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plots and self.checkpoint_base_dir is not None:
            checkpoint_dir = Path(self.checkpoint_base_dir) / "average_parameters"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = int(time.time())
            strategy_name = self.parameter_strategy.__class__.__name__
            
            plot_filename = f"baseline_{strategy_name}_average_parameters_{len(function_names)}_functions_{timestamp}.png"
            plot_path = checkpoint_dir / plot_filename
            
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            log_success(f"Average parameter plot saved to {plot_path}", self.module_name)
        
        # Show plot if requested
        if show_plots:
            plt.show()
        
        plt.close()

    def plot_stability_condition(self, function_names=None, save_plots=True, show_plots=False):
        """
        Plot c1+c2 and the stability RHS for each function and averaged across all functions.
        Also plot the fraction of particles that are stable based on the stability condition.
        """
        if not hasattr(self, 'parameter_tracking') or not self.parameter_tracking:
            log_warning("No parameter tracking data available. Run PSO evaluation first.", self.module_name)
            return

        if function_names is None:
            function_names = list(self.parameter_tracking.keys())

        steps = np.arange(self.env_max_steps)

        # Per-function plots
        for func_name in function_names:
            data = self.parameter_tracking[func_name]
            c1 = np.array(data['c1'])
            c2 = np.array(data['c2'])
            omega = np.array(data['omega'])
            lhs = np.mean(c1 + c2, axis=0)
            rhs = np.mean(24 * (1 - omega**2) / (7 - 5*omega), axis=0)
            
            # Calculate stability fraction for each run and step
            stability_fractions = []
            for run_idx in range(len(c1)):
                run_stability = []
                for step_idx in range(len(c1[run_idx])):
                    # Check if stability condition is satisfied
                    if (-1.0 <= omega[run_idx][step_idx] <= 1.0):
                        denominator = 7.0 - 5.0 * omega[run_idx][step_idx]
                        if not np.isclose(denominator, 0):
                            stability_boundary = 24.0 * (1.0 - omega[run_idx][step_idx]**2) / denominator
                            is_stable = (c1[run_idx][step_idx] + c2[run_idx][step_idx]) < stability_boundary
                            run_stability.append(1.0 if is_stable else 0.0)
                        else:
                            run_stability.append(0.0)  # Unstable if denominator is zero
                    else:
                        run_stability.append(0.0)  # Unstable if omega is out of range
                stability_fractions.append(run_stability)
            
            # Calculate average stability fraction across runs
            stability_fractions = np.array(stability_fractions)
            avg_stability_fraction = np.mean(stability_fractions, axis=0)

            # Create subplots: one for stability condition, one for stability fraction
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
            ax2.plot(steps, avg_stability_fraction, label='Fraction of Stable Particles', color='red', linewidth=2)
            ax2.set_title(f"Fraction of Stable Particles: {func_name}")
            ax2.set_xlabel("Optimization Step")
            ax2.set_ylabel("Stability Fraction")
            ax2.set_ylim(0, 1)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add statistics text box
            final_stability = avg_stability_fraction[-1]
            stats_text = f'Final Stability: {final_stability:.3f}\nMean Stability: {np.mean(avg_stability_fraction):.3f}'
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            if save_plots and self.checkpoint_base_dir is not None:
                checkpoint_dir = Path(self.checkpoint_base_dir) / "stability_condition"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(checkpoint_dir / f"stability_condition_{func_name}.png", dpi=300, bbox_inches='tight')
                log_success(f"Stability condition plot saved for {func_name}", self.module_name)
            if show_plots:
                plt.show()
            plt.close()

        # Averaged across all functions
        all_c1, all_c2, all_omega = [], [], []
        for func_name in function_names:
            data = self.parameter_tracking[func_name]
            all_c1.extend(data['c1'])
            all_c2.extend(data['c2'])
            all_omega.extend(data['omega'])
        all_c1, all_c2, all_omega = np.array(all_c1), np.array(all_c2), np.array(all_omega)
        lhs_avg = np.mean(all_c1 + all_c2, axis=0)
        rhs_avg = np.mean(24 * (1 - all_omega**2) / (7 - 5*all_omega), axis=0)
        
        # Calculate overall stability fraction
        overall_stability_fractions = []
        for run_idx in range(len(all_c1)):
            run_stability = []
            for step_idx in range(len(all_c1[run_idx])):
                if (-1.0 <= all_omega[run_idx][step_idx] <= 1.0):
                    denominator = 7.0 - 5.0 * all_omega[run_idx][step_idx]
                    if not np.isclose(denominator, 0):
                        stability_boundary = 24.0 * (1.0 - all_omega[run_idx][step_idx]**2) / denominator
                        is_stable = (all_c1[run_idx][step_idx] + all_c2[run_idx][step_idx]) < stability_boundary
                        run_stability.append(1.0 if is_stable else 0.0)
                    else:
                        run_stability.append(0.0)
                else:
                    run_stability.append(0.0)
            overall_stability_fractions.append(run_stability)
        
        overall_stability_fractions = np.array(overall_stability_fractions)
        avg_overall_stability = np.mean(overall_stability_fractions, axis=0)

        # Create subplots for averaged results
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Average stability condition
        ax1.plot(steps, lhs_avg, label=r"Average $c_1 + c_2$", color='blue', linewidth=2)
        ax1.plot(steps, rhs_avg, label=r"Average $\frac{24(1-w^2)}{7-5w}$", color='orange', linewidth=2)
        ax1.set_title("Stability Condition (Averaged Across All Functions)")
        ax1.set_xlabel("Optimization Step")
        ax1.set_ylabel("Value")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add stability region shading
        ax1.fill_between(steps, 0, rhs_avg, alpha=0.2, color='green', label='Stable Region')
        
        # Plot 2: Average stability fraction
        ax2.plot(steps, avg_overall_stability, label='Average Fraction of Stable Particles', color='red', linewidth=2)
        ax2.set_title("Fraction of Stable Particles (Averaged Across All Functions)")
        ax2.set_xlabel("Optimization Step")
        ax2.set_ylabel("Stability Fraction")
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text box
        final_stability = avg_overall_stability[-1]
        stats_text = f'Final Stability: {final_stability:.3f}\nMean Stability: {np.mean(avg_overall_stability):.3f}'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plots and self.checkpoint_base_dir is not None:
            checkpoint_dir = Path(self.checkpoint_base_dir) / "stability_condition"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(checkpoint_dir / f"stability_condition_average.png", dpi=300, bbox_inches='tight')
            log_success("Average stability condition plot saved", self.module_name)
        if show_plots:
            plt.show()
        plt.close()

    def plot_infeasible_particles(self, function_names=None, save_plots=True, show_plots=False):
        """
        Plot the fraction of infeasible particles over optimization steps for each function.
        
        Args:
            function_names: List of function names to plot. If None, plots all tracked functions.
            save_plots: Whether to save the plots to files
            show_plots: Whether to display the plots
        """
        if not hasattr(self, 'parameter_tracking') or not self.parameter_tracking:
            log_warning("No parameter tracking data available. Run PSO evaluation first.", self.module_name)
            return

        if function_names is None:
            function_names = list(self.parameter_tracking.keys())

        log_info(f"Creating infeasible particles plots for {len(function_names)} functions", self.module_name)

        steps = np.arange(self.env_max_steps)

        # Per-function plots
        for func_name in function_names:
            if func_name not in self.parameter_tracking:
                log_warning(f"No parameter data for function {func_name}", self.module_name)
                continue

            data = self.parameter_tracking[func_name]
            
            # Check if infeasible ratios data exists
            if 'infeasible_ratios' not in data:
                log_warning(f"No infeasible ratios data for function {func_name}. Skipping plot.", self.module_name)
                continue
            
            # Calculate average infeasible ratio across all runs
            infeasible_data = np.array(data['infeasible_ratios'])
            avg_infeasible_fraction = np.mean(infeasible_data, axis=0)
            std_infeasible_fraction = np.std(infeasible_data, axis=0)
            
            # Create plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # Plot average infeasible fraction with error bands
            ax.plot(steps, avg_infeasible_fraction, label='Average Fraction of Infeasible Particles', 
                    color='red', linewidth=2)
            ax.fill_between(steps, 
                           avg_infeasible_fraction - std_infeasible_fraction,
                           avg_infeasible_fraction + std_infeasible_fraction,
                           alpha=0.3, color='red', label='±1 Std Dev')
            
            ax.set_title(f"Infeasible Particles: {func_name}")
            ax.set_xlabel("Optimization Step")
            ax.set_ylabel("Infeasible Fraction")
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics text box
            final_infeasible = avg_infeasible_fraction[-1]
            mean_infeasible = np.mean(avg_infeasible_fraction)
            max_infeasible = np.max(avg_infeasible_fraction)
            stats_text = f'Final Infeasible: {final_infeasible:.3f}\nMean Infeasible: {mean_infeasible:.3f}\nMax Infeasible: {max_infeasible:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            if save_plots and self.checkpoint_base_dir is not None:
                checkpoint_dir = Path(self.checkpoint_base_dir) / "infeasible_particles"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(checkpoint_dir / f"infeasible_particles_{func_name}.png", dpi=300, bbox_inches='tight')
                log_success(f"Infeasible particles plot saved for {func_name}", self.module_name)
            if show_plots:
                plt.show()
            plt.close()

        # Averaged across all functions
        all_infeasible_data = []
        for func_name in function_names:
            if func_name in self.parameter_tracking and 'infeasible_ratios' in self.parameter_tracking[func_name]:
                data = self.parameter_tracking[func_name]
                all_infeasible_data.extend(data['infeasible_ratios'])
        
        if all_infeasible_data:
            all_infeasible_data = np.array(all_infeasible_data)
            avg_overall_infeasible = np.mean(all_infeasible_data, axis=0)
            std_overall_infeasible = np.std(all_infeasible_data, axis=0)

            # Create averaged plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            ax.plot(steps, avg_overall_infeasible, label='Average Fraction of Infeasible Particles', 
                    color='red', linewidth=2)
            ax.fill_between(steps, 
                           avg_overall_infeasible - std_overall_infeasible,
                           avg_overall_infeasible + std_overall_infeasible,
                           alpha=0.3, color='red', label='±1 Std Dev')
            ax.set_title("Infeasible Particles (Averaged Across All Functions)")
            ax.set_xlabel("Optimization Step")
            ax.set_ylabel("Infeasible Fraction")
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics text box
            final_infeasible = avg_overall_infeasible[-1]
            mean_infeasible = np.mean(avg_overall_infeasible)
            max_infeasible = np.max(avg_overall_infeasible)
            stats_text = f'Final Infeasible: {final_infeasible:.3f}\nMean Infeasible: {mean_infeasible:.3f}\nMax Infeasible: {max_infeasible:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            if save_plots and self.checkpoint_base_dir is not None:
                checkpoint_dir = Path(self.checkpoint_base_dir) / "infeasible_particles"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(checkpoint_dir / f"infeasible_particles_average.png", dpi=300, bbox_inches='tight')
                log_success("Average infeasible particles plot saved", self.module_name)
            if show_plots:
                plt.show()
            plt.close()

    def plot_average_velocity(self, function_names=None, save_plots=True, show_plots=False):
        """
        Plot the average velocity magnitude of particles over optimization steps for each function.
        Uses logarithmic scale for better visualization of velocity changes.
        
        Args:
            function_names: List of function names to plot. If None, plots all tracked functions.
            save_plots: Whether to save the plots to files
            show_plots: Whether to display the plots
        """
        if not hasattr(self, 'parameter_tracking') or not self.parameter_tracking:
            log_warning("No parameter tracking data available. Run PSO evaluation first.", self.module_name)
            return

        if function_names is None:
            function_names = list(self.parameter_tracking.keys())

        log_info(f"Creating average velocity plots for {len(function_names)} functions", self.module_name)

        steps = np.arange(self.env_max_steps)

        # Per-function plots
        for func_name in function_names:
            if func_name not in self.parameter_tracking:
                log_warning(f"No parameter data for function {func_name}", self.module_name)
                continue

            data = self.parameter_tracking[func_name]
            
            # Check if velocity data exists
            if 'avg_velocities' not in data:
                log_warning(f"No velocity data for function {func_name}. Skipping plot.", self.module_name)
                continue
            
            # Calculate average velocity across all runs
            velocity_data = np.array(data['avg_velocities'])
            avg_velocity = np.mean(velocity_data, axis=0)
            std_velocity = np.std(velocity_data, axis=0)
            
            # Create plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # Plot average velocity with error bands on log scale
            ax.semilogy(steps, avg_velocity, label='Average Velocity Magnitude', 
                       color='blue', linewidth=2)
            
            # Add error bands (log scale)
            ax.fill_between(steps, 
                           np.maximum(avg_velocity - std_velocity, 1e-10),  # Avoid negative values for log scale
                           avg_velocity + std_velocity,
                           alpha=0.3, color='blue', label='±1 Std Dev')
            
            ax.set_title(f"Average Particle Velocity: {func_name}")
            ax.set_xlabel("Optimization Step")
            ax.set_ylabel("Average Velocity Magnitude (log scale)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics text box
            final_velocity = avg_velocity[-1]
            mean_velocity = np.mean(avg_velocity)
            max_velocity = np.max(avg_velocity)
            min_velocity = np.min(avg_velocity)
            stats_text = f'Final Velocity: {final_velocity:.2e}\nMean Velocity: {mean_velocity:.2e}\nMax Velocity: {max_velocity:.2e}\nMin Velocity: {min_velocity:.2e}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            
            if save_plots and self.checkpoint_base_dir is not None:
                checkpoint_dir = Path(self.checkpoint_base_dir) / "average_velocity"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(checkpoint_dir / f"average_velocity_{func_name}.png", dpi=300, bbox_inches='tight')
                log_success(f"Average velocity plot saved for {func_name}", self.module_name)
            if show_plots:
                plt.show()
            plt.close()

        # Averaged across all functions
        all_velocity_data = []
        for func_name in function_names:
            if func_name in self.parameter_tracking and 'avg_velocities' in self.parameter_tracking[func_name]:
                data = self.parameter_tracking[func_name]
                all_velocity_data.extend(data['avg_velocities'])
        
        if all_velocity_data:
            all_velocity_data = np.array(all_velocity_data)
            avg_overall_velocity = np.mean(all_velocity_data, axis=0)
            std_overall_velocity = np.std(all_velocity_data, axis=0)

            # Create averaged plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            ax.semilogy(steps, avg_overall_velocity, label='Average Velocity Magnitude', 
                       color='blue', linewidth=2)
            ax.fill_between(steps, 
                           np.maximum(avg_overall_velocity - std_overall_velocity, 1e-10),
                           avg_overall_velocity + std_overall_velocity,
                           alpha=0.3, color='blue', label='±1 Std Dev')
            ax.set_title("Average Particle Velocity (Averaged Across All Functions)")
            ax.set_xlabel("Optimization Step")
            ax.set_ylabel("Average Velocity Magnitude (log scale)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics text box
            final_velocity = avg_overall_velocity[-1]
            mean_velocity = np.mean(avg_overall_velocity)
            max_velocity = np.max(avg_overall_velocity)
            min_velocity = np.min(avg_overall_velocity)
            stats_text = f'Final Velocity: {final_velocity:.2e}\nMean Velocity: {mean_velocity:.2e}\nMax Velocity: {max_velocity:.2e}\nMin Velocity: {min_velocity:.2e}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            
            if save_plots and self.checkpoint_base_dir is not None:
                checkpoint_dir = Path(self.checkpoint_base_dir) / "average_velocity"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(checkpoint_dir / f"average_velocity_average.png", dpi=300, bbox_inches='tight')
                log_success("Average velocity plot saved", self.module_name)
            if show_plots:
                plt.show()
            plt.close()

    def plot_swarm_diversity(self, diversities, function_name, save_plots=True, show_plots=False, clip_std_dev=False):
        """
        Plot the swarm diversity over steps.
        Args:
            diversities: List of lists of diversity values (one list per run)
            function_name: Name of the function being optimized
            save_plots: Whether to save the plot
            show_plots: Whether to display the plot
            clip_std_dev: Whether to clip the standard deviation bands to avoid extreme values
        """
        try:
            avg_diversity = np.mean(diversities, axis=0)
            std_diversity = np.std(diversities, axis=0)
            steps = np.arange(len(avg_diversity))

            plt.figure(figsize=(10, 6))
            plt.semilogy(steps, avg_diversity, label='Average Swarm Diversity', color='green', linewidth=2)
            
            if not clip_std_dev:
                plt.fill_between(steps, 
                               np.maximum(avg_diversity - std_diversity, 1e-10),  # Avoid negative values for log scale
                               avg_diversity + std_diversity,
                               alpha=0.3, color='green', label='±1 Std Dev')
            
            plt.xlabel('Optimization Step')
            plt.ylabel('Swarm Diversity (log scale)')
            title_suffix = " (Clipped)" if clip_std_dev else ""
            plt.title(f'Swarm Diversity over Steps ({function_name}){title_suffix}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add statistics text box
            final_diversity = avg_diversity[-1]
            mean_diversity = np.mean(avg_diversity)
            max_diversity = np.max(avg_diversity)
            min_diversity = np.min(avg_diversity)
            stats_text = f'Final Diversity: {final_diversity:.4f}\nMean Diversity: {mean_diversity:.4f}\nMax Diversity: {max_diversity:.4f}\nMin Diversity: {min_diversity:.4f}'
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            plt.tight_layout()
            
            if save_plots and self.checkpoint_base_dir is not None:
                checkpoint_dir = Path(self.checkpoint_base_dir) / "swarm_diversity"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(checkpoint_dir / f"swarm_diversity_{function_name}.png", dpi=300, bbox_inches='tight')
                log_success(f"Swarm diversity plot saved for {function_name}", self.module_name)
            elif save_plots:
                plt.savefig(f'baseline_swarm_diversity_{function_name}.png', dpi=300, bbox_inches='tight')
                log_success(f"Swarm diversity plot saved for {function_name}", self.module_name)
            
            if show_plots:
                plt.show()
            plt.close()
            
        except Exception as e:
            log_error(f"Error creating swarm diversity plot for {function_name}: {e}", self.module_name)
            plt.close()

    def plot_average_swarm_diversity(self, save_plots=True, show_plots=False, clip_std_dev=False):
        """
        Plot the average swarm diversity across all functions.
        Args:
            save_plots: Whether to save the plot
            show_plots: Whether to display the plot
            clip_std_dev: Whether to clip the standard deviation bands to avoid extreme values
        """
        try:
            if not hasattr(self, 'diversity_results') or not self.diversity_results:
                log_warning("No diversity data available. Run PSO evaluation first.", self.module_name)
                return
            
            # Collect all diversity data
            all_diversity_data = []
            for func_name, diversities in self.diversity_results.items():
                if diversities and len(diversities) > 0:
                    all_diversity_data.extend(diversities)
            
            if not all_diversity_data:
                log_warning("No valid diversity data found.", self.module_name)
                return
            
            # Calculate average across all functions
            all_diversity_data = np.array(all_diversity_data)
            avg_overall_diversity = np.mean(all_diversity_data, axis=0)
            std_overall_diversity = np.std(all_diversity_data, axis=0)
            steps = np.arange(len(avg_overall_diversity))

            # Create averaged plot
            plt.figure(figsize=(10, 6))
            plt.semilogy(steps, avg_overall_diversity, label='Average Swarm Diversity', 
                    color='green', linewidth=2)
            
            if not clip_std_dev:
                plt.fill_between(steps, 
                               np.maximum(avg_overall_diversity - std_overall_diversity, 1e-10),  # Avoid negative values for log scale
                               avg_overall_diversity + std_overall_diversity,
                               alpha=0.3, color='green', label='±1 Std Dev')
            
            plt.xlabel('Optimization Step')
            plt.ylabel('Swarm Diversity (log scale)')
            title_suffix = " (Clipped)" if clip_std_dev else ""
            plt.title(f'Average Swarm Diversity (Averaged Across All Functions){title_suffix}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add statistics text box
            final_diversity = avg_overall_diversity[-1]
            mean_diversity = np.mean(avg_overall_diversity)
            max_diversity = np.max(avg_overall_diversity)
            min_diversity = np.min(avg_overall_diversity)
            stats_text = f'Final Diversity: {final_diversity:.4f}\nMean Diversity: {mean_diversity:.4f}\nMax Diversity: {max_diversity:.4f}\nMin Diversity: {min_diversity:.4f}'
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            plt.tight_layout()
            
            if save_plots and self.checkpoint_base_dir is not None:
                checkpoint_dir = Path(self.checkpoint_base_dir) / "swarm_diversity"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(checkpoint_dir / f"swarm_diversity_average.png", dpi=300, bbox_inches='tight')
                log_success("Average swarm diversity plot saved", self.module_name)
            elif save_plots:
                plt.savefig(f'baseline_swarm_diversity_average.png', dpi=300, bbox_inches='tight')
                log_success("Average swarm diversity plot saved", self.module_name)
            
            if show_plots:
                plt.show()
            plt.close()
            
        except Exception as e:
            log_error(f"Error creating average swarm diversity plot: {e}", self.module_name)
            plt.close()


def run_baseline_pso(env_dim: int = ENV_DIM,
                    env_particles: int = ENV_PARTICLES,
                    env_max_steps: int = ENV_MAX_STEPS,
                    episodes_per_function: int = EPISODES_PER_FUNCTION,
                    num_eval_runs: int = NUM_EVAL_RUNS,
                    max_training_functions: Optional[int] = None,
                    checkpoint_base_dir: Optional[str] = CHECKPOINT_BASE_DIR,
                    parameter_strategy: Union[str, ParameterStrategy] = 'fixed',
                    strategy_kwargs: Optional[Dict] = None):
    """
    Convenience function to run the complete baseline PSO evaluation.
    
    Args:
        env_dim: Problem dimension
        env_particles: Number of particles in the swarm
        env_max_steps: Maximum PSO steps per run
        episodes_per_function: Number of runs per training function
        num_eval_runs: Number of runs per testing function
        max_training_functions: Maximum number of training functions to test
        checkpoint_base_dir: Base directory for saving results
        parameter_strategy: Parameter strategy to use (string name or ParameterStrategy instance)
        strategy_kwargs: Additional keyword arguments for the parameter strategy
        
    Returns:
        Dictionary containing both training and testing results
    """
    baseline = BaselinePSO(
        env_dim=env_dim,
        env_particles=env_particles,
        env_max_steps=env_max_steps,
        checkpoint_base_dir=checkpoint_base_dir,
        parameter_strategy=parameter_strategy,
        strategy_kwargs=strategy_kwargs
    )
    
    return baseline.run_full_baseline(
        episodes_per_function=episodes_per_function,
        num_eval_runs=num_eval_runs,
        max_training_functions=max_training_functions
    )


def list_available_strategies() -> Dict[str, str]:
    """
    Get a list of available parameter strategies with descriptions.
    
    Returns:
        Dictionary mapping strategy names to descriptions
    """
    from Baseline.parameter_strategies import list_available_strategies as _list_strategies
    return _list_strategies()


if __name__ == "__main__":
    log_header("Starting Baseline PSO Evaluation", "main")

    # Create BaselinePSO instance
    baseline = BaselinePSO(
        env_dim=ENV_DIM,
        env_particles=ENV_PARTICLES,
        env_max_steps=ENV_MAX_STEPS,
        checkpoint_base_dir=CHECKPOINT_BASE_DIR,
        parameter_strategy='linear_decay',
        strategy_kwargs={'omega_max': 0.9, 'omega_min': 0.4}
    )

    # Run experiments
    results = baseline.run_full_baseline(
        episodes_per_function=5,  # Reduced for faster testing
        num_eval_runs=5,  # Reduced for faster testing
        max_training_functions=3  # Test only first 3 functions
    )

    log_header("Baseline PSO Evaluation Complete", "main")