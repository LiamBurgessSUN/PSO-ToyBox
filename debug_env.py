#!/usr/bin/env python3
"""
Debug script for testing the PSO environment.
This script allows you to debug the environment without running the full training pipeline.
"""

import numpy as np
from SAPSO_AGENT.SAPSO.Environment.Environment import Environment
from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.Training.Functions.Ackley import AckleyFunction
from SAPSO_AGENT.Logs.logger import log_info, log_header

def debug_environment():
    """Debug the PSO environment step by step."""
    log_header("=== PSO Environment Debug ===", "debug_env")
    
    # Create a simple objective function
    obj_func = AckleyFunction(dim=2)  # Use 2D for easier debugging
    log_info(f"Created objective function: {obj_func.__class__.__name__}", "debug_env")
    
    # Create environment
    env = Environment(
        obj_func=obj_func,
        num_particles=5,  # Small swarm for debugging
        max_steps=100,
        agent_step_size=10,
        adaptive_nt=False,
        nt_range=(1, 50),
        v_clamp_ratio=0.2,
        use_velocity_clamping=True,
        convergence_patience=10,
        convergence_threshold_gbest=1e-6,
        convergence_threshold_pbest_std=1e-6
    )
    log_info("Environment created successfully", "debug_env")
    
    # Reset environment
    state, info = env.reset()
    log_info(f"Environment reset. Initial state: {state}", "debug_env")
    log_info(f"Initial gbest: {env.pso.gbest_value}", "debug_env")
    
    # Run a few steps
    for step in range(5):
        # Generate a random action
        action = np.random.uniform(-1, 1, env.action_space.shape[0])
        log_info(f"Step {step + 1}: Action = {action}", "debug_env")
        
        # Take step
        next_state, reward, terminated, truncated, info = env.step(action)
        log_info(f"Step {step + 1}: Reward = {reward:.4f}, GBest = {env.pso.gbest_value:.6f}", "debug_env")
        log_info(f"Step {step + 1}: State = {next_state}", "debug_env")
        
        if terminated or truncated:
            log_info(f"Episode ended at step {step + 1}", "debug_env")
            break
    
    log_info("Environment debug completed", "debug_env")

if __name__ == "__main__":
    debug_environment() 