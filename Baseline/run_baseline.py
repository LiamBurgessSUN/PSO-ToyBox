#!/usr/bin/env python3
"""
Baseline PSO Runner Script

This script provides a simple interface to run the baseline PSO evaluation
without any reinforcement learning components, with support for different
parameter update strategies.
"""

import sys
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import from SAPSO_AGENT
sys.path.append(str(Path(__file__).parent.parent))

from Baseline.baseline_pso import run_baseline_pso, list_available_strategies, BaselinePSO
from SAPSO_AGENT.CONFIG import *
from SAPSO_AGENT.Logs.logger import log_header, log_info, log_warning


def main():
    """Main function to run baseline PSO evaluation."""
    parser = argparse.ArgumentParser(description='Run baseline PSO evaluation with different parameter strategies')
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'testing', 'both'], default='both',
                       help='Mode to run: train (training functions only), test/testing (testing functions only), or both (default: both)')
    
    # Basic configuration
    parser.add_argument('--strategy', type=str, default='fixed',
                       help='Parameter strategy to use (default: fixed)')
    parser.add_argument('--env-dim', type=int, default=ENV_DIM,
                       help=f'Problem dimension (default: {ENV_DIM})')
    parser.add_argument('--particles', type=int, default=ENV_PARTICLES,
                       help=f'Number of particles (default: {ENV_PARTICLES})')
    parser.add_argument('--max-steps', type=int, default=ENV_MAX_STEPS,
                       help=f'Maximum PSO steps (default: {ENV_MAX_STEPS})')
    
    # Evaluation parameters
    parser.add_argument('--episodes-per-function', type=int, default=EPISODES_PER_FUNCTION,
                       help=f'Episodes per training function (default: {EPISODES_PER_FUNCTION})')
    parser.add_argument('--eval-runs', type=int, default=NUM_EVAL_RUNS,
                       help=f'Evaluation runs per test function (default: {NUM_EVAL_RUNS})')
    parser.add_argument('--max-training-functions', type=int, default=None,
                       help='Maximum number of training functions to test (default: all)')
    
    # Strategy-specific parameters
    parser.add_argument('--omega', type=float, default=0.7,
                       help='Inertia weight (for fixed strategy, default: 0.7)')
    parser.add_argument('--c1', type=float, default=1.5,
                       help='Cognitive coefficient (for fixed strategy, default: 1.5)')
    parser.add_argument('--c2', type=float, default=1.5,
                       help='Social coefficient (for fixed strategy, default: 1.5)')
    parser.add_argument('--omega-max', type=float, default=0.9,
                       help='Maximum inertia weight (for decay strategies, default: 0.9)')
    parser.add_argument('--omega-min', type=float, default=0.4,
                       help='Minimum inertia weight (for decay strategies, default: 0.4)')
    
    # Output
    parser.add_argument('--checkpoint-dir', type=str, default=CHECKPOINT_BASE_DIR,
                       help=f'Checkpoint directory (default: {CHECKPOINT_BASE_DIR})')
    parser.add_argument('--list-strategies', action='store_true',
                       help='List available parameter strategies and exit')
    
    # Plotting options
    parser.add_argument('--plot-parameters', action='store_true',
                       help='Generate parameter evolution plots after evaluation')
    parser.add_argument('--plot-parameter-evolution', action='store_true',
                       help='Generate parameter evolution plots after evaluation (alias for --plot-parameters)')
    parser.add_argument('--plot-comparison', action='store_true',
                       help='Generate parameter comparison plots across functions')
    parser.add_argument('--plot-parameter-comparison', action='store_true',
                       help='Generate parameter comparison plots across functions (alias for --plot-comparison)')
    parser.add_argument('--plot-average', action='store_true',
                       help='Generate single plot with average parameters across all functions')
    parser.add_argument('--plot-average-parameters', action='store_true',
                       help='Generate single plot with average parameters across all functions (alias for --plot-average)')
    parser.add_argument('--plot-stability', action='store_true',
                       help='Generate stability condition plots')
    parser.add_argument('--show-plots', action='store_true',
                       help='Display plots in addition to saving them')
    parser.add_argument('--plot-functions', type=str, nargs='+',
                       help='Specific functions to include in plots (default: all)')
    parser.add_argument('--plot-infeasible', action='store_true',
                       help='Generate infeasible particles plots after evaluation')
    parser.add_argument('--plot-velocity', action='store_true',
                       help='Generate average velocity plots after evaluation')
    parser.add_argument('--plot-all', action='store_true',
                       help='Generate all plots after evaluation')
    parser.add_argument('--plot_swarm_diversity', action='store_true', help='Plot swarm diversity for each function')
    parser.add_argument('--clip_swarm_diversity', action='store_true', help='Clip standard deviation bands in swarm diversity plots to focus on average')
    
    args = parser.parse_args()
    
    if args.list_strategies:
        print("Available parameter strategies:")
        strategies = list_available_strategies()
        for name, description in strategies.items():
            print(f"  {name}: {description}")
        return 0
    
    log_header("Baseline PSO Evaluation Runner", "main")
    
    # Configuration
    log_info("Configuration:", "main")
    log_info(f"  Mode: {args.mode}", "main")
    log_info(f"  Parameter Strategy: {args.strategy}", "main")
    log_info(f"  Environment Dimension: {args.env_dim}", "main")
    log_info(f"  Number of Particles: {args.particles}", "main")
    log_info(f"  Maximum Steps: {args.max_steps}", "main")
    log_info(f"  Episodes per Function: {args.episodes_per_function}", "main")
    log_info(f"  Evaluation Runs per Function: {args.eval_runs}", "main")
    log_info(f"  Checkpoint Directory: {args.checkpoint_dir}", "main")
    
    # Build strategy kwargs based on strategy type
    strategy_kwargs = {}
    
    if args.strategy == 'fixed':
        strategy_kwargs = {
            'omega': args.omega,
            'c1': args.c1,
            'c2': args.c2
        }
    elif args.strategy in ['linear_decay', 'exponential_decay']:
        strategy_kwargs = {
            'omega_max': args.omega_max,
            'omega_min': args.omega_min,
            'c1': args.c1,
            'c2': args.c2
        }
    elif args.strategy == 'adaptive_cognitive_social':
        strategy_kwargs = {
            'omega': args.omega,
            'c1_max': 2.5,
            'c1_min': 0.5,
            'c2_min': 0.5,
            'c2_max': 2.5
        }
    elif args.strategy == 'chaos_based':
        strategy_kwargs = {
            'omega_base': args.omega,
            'c1_base': args.c1,
            'c2_base': args.c2,
            'omega_amplitude': 0.3,
            'c1_amplitude': 0.5,
            'c2_amplitude': 0.5
        }
    elif args.strategy == 'fitness_based':
        strategy_kwargs = {
            'omega_base': args.omega,
            'c1_base': args.c1,
            'c2_base': args.c2,
            'improvement_threshold': 1e-6
        }
    elif args.strategy == 'time_varying':
        strategy_kwargs = {
            'omega_center': args.omega,
            'c1_center': args.c1,
            'c2_center': args.c2,
            'omega_amplitude': 0.2,
            'c1_amplitude': 0.3,
            'c2_amplitude': 0.3,
            'frequency': 1.0
        }
    elif args.strategy == 'paper_time_varying':
        strategy_kwargs = {
        }
    
    log_info(f"  Strategy Parameters: {strategy_kwargs}", "main")
    
    # Create BaselinePSO instance
    baseline_pso = BaselinePSO(
        env_dim=args.env_dim,
        env_particles=args.particles,
        env_max_steps=args.max_steps,
        checkpoint_base_dir=args.checkpoint_dir,
        parameter_strategy=args.strategy,
        strategy_kwargs=strategy_kwargs
    )
    
    results = {"training": {}, "testing": {}}
    
    # Run baseline PSO evaluation based on mode
    try:
        if args.mode in ['train', 'both']:
            log_header("Running Training Functions", "main")
            training_results = baseline_pso.run_training_functions(
                episodes_per_function=args.episodes_per_function,
                max_functions=args.max_training_functions
            )
            results["training"] = training_results
            
            # Save training results
            baseline_pso.training_results = training_results
            baseline_pso.save_results("training")
            
            log_info(f"Training functions evaluated: {len(training_results)}", "main")
            
            # Calculate training statistics
            all_training_gbests = []
            for func_results in training_results.values():
                all_training_gbests.extend([g for g, s in func_results if g != float('inf')])
            
            if all_training_gbests:
                log_info(f"Training - Mean GBest: {sum(all_training_gbests)/len(all_training_gbests):.6e}", "main")
        
        if args.mode in ['test', 'testing', 'both']:
            log_header("Running Testing Functions", "main")
            testing_results = baseline_pso.run_testing_functions(
                num_eval_runs=args.eval_runs
            )
            results["testing"] = testing_results
            
            # Save testing results
            baseline_pso.testing_results = testing_results
            baseline_pso.save_results("testing")
            
            log_info(f"Testing functions evaluated: {len(testing_results)}", "main")
            
            # Calculate testing statistics
            all_testing_gbests = []
            for func_results in testing_results.values():
                all_testing_gbests.extend([g for g, s in func_results if g != float('inf')])
            
            if all_testing_gbests:
                log_info(f"Testing - Mean GBest: {sum(all_testing_gbests)/len(all_testing_gbests):.6e}", "main")
        
        log_header("Baseline PSO Evaluation Completed Successfully", "main")
        
        # Generate plots if requested
        if (args.plot_parameters or args.plot_parameter_evolution or args.plot_comparison or 
            args.plot_parameter_comparison or args.plot_average or args.plot_average_parameters or 
            args.plot_stability or args.plot_infeasible or args.plot_velocity or args.plot_swarm_diversity or args.plot_all):
            log_header("Generating Parameter Plots", "main")
            
            # Determine which functions to plot
            plot_functions = args.plot_functions if args.plot_functions else None
            
            # Handle plot-all flag
            if args.plot_all:
                args.plot_parameters = True
                args.plot_comparison = True
                args.plot_average = True
                args.plot_stability = True
                args.plot_infeasible = True
                args.plot_velocity = True
                args.plot_swarm_diversity = True
            
            # Handle alias arguments
            if args.plot_parameter_evolution:
                args.plot_parameters = True
            if args.plot_parameter_comparison:
                args.plot_comparison = True
            if args.plot_average_parameters:
                args.plot_average = True
            
            if args.plot_parameters:
                log_info("Creating parameter evolution plots...", "main")
                baseline_pso.plot_parameter_evolution(
                    function_names=plot_functions,
                    save_plots=True,
                    show_plots=args.show_plots
                )
            
            if args.plot_comparison:
                log_info("Creating parameter comparison plots...", "main")
                baseline_pso.plot_parameter_comparison(
                    function_names=plot_functions,
                    save_plots=True,
                    show_plots=args.show_plots
                )
            
            if args.plot_average:
                log_info("Creating average parameter plot...", "main")
                baseline_pso.plot_average_parameters(
                    function_names=plot_functions,
                    save_plots=True,
                    show_plots=args.show_plots
                )
            
            if args.plot_stability:
                log_info("Creating stability condition plots...", "main")
                baseline_pso.plot_stability_condition(
                    function_names=plot_functions,
                    save_plots=True,
                    show_plots=args.show_plots
                )
            
            if args.plot_infeasible:
                log_info("Creating infeasible particles plots...", "main")
                baseline_pso.plot_infeasible_particles(
                    function_names=plot_functions,
                    save_plots=True,
                    show_plots=args.show_plots
                )
            
            if args.plot_velocity:
                log_info("Creating average velocity plots...", "main")
                baseline_pso.plot_average_velocity(
                    function_names=plot_functions,
                    save_plots=True,
                    show_plots=args.show_plots
                )
            
            if args.plot_swarm_diversity:
                log_info("Creating swarm diversity plots...", "main")
                for func_name, diversities in baseline_pso.diversity_results.items():
                    baseline_pso.plot_swarm_diversity(diversities, func_name, clip_std_dev=args.clip_swarm_diversity)
                
                # Create average plot across all functions
                baseline_pso.plot_average_swarm_diversity(clip_std_dev=args.clip_swarm_diversity)
    except Exception as e:
        log_info(f"Error during baseline PSO evaluation: {e}", "main")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 