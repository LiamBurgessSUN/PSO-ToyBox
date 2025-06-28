#!/usr/bin/env python3
"""
Test script for the vectorized PSO implementation.
This script tests the core PSO functionality without the RL components.
"""

import numpy as np
from SAPSO_AGENT.SAPSO.PSO.PSO import PSOVectorized
from SAPSO_AGENT.SAPSO.PSO.Cognitive.GBest import GlobalBestStrategy
from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.Training.Functions.Rastrgin import RastriginFunction
from SAPSO_AGENT.Logs.logger import log_info, log_header, log_success

def test_pso_vectorized():
    """Test the vectorized PSO implementation."""
    log_header("=== PSO Vectorized Test ===", "test_pso")
    
    # Create objective function
    obj_func = RastriginFunction(dim=10)
    log_info(f"Created objective function: {obj_func.__class__.__name__}", "test_pso")
    
    # Create strategy
    strategy = GlobalBestStrategy(None)
    log_info("Created GlobalBestStrategy", "test_pso")
    
    # Create PSO
    pso = PSOVectorized(
        objective_function=obj_func,
        num_particles=20,
        strategy=strategy,
        v_clamp_ratio=0.2,
        use_velocity_clamping=True,
        convergence_patience=10,
        convergence_threshold_gbest=1e-6,
        convergence_threshold_pbest_std=1e-6
    )
    strategy.swarm = pso
    log_info("Created PSOVectorized instance", "test_pso")
    
    # Test initial state
    log_info(f"Initial gbest value: {pso.gbest_value:.6f}", "test_pso")
    log_info(f"Initial gbest position: {pso.gbest_position[:5]}...", "test_pso")
    log_info(f"Number of particles: {pso.num_particles}", "test_pso")
    log_info(f"Problem dimension: {pso.dim}", "test_pso")
    
    # Run optimization steps
    log_info("Starting optimization...", "test_pso")
    for step in range(50):
        # Use some reasonable PSO parameters
        omega = 0.7
        c1 = 1.5
        c2 = 1.5
        
        # Optimize one step
        metrics, gbest_value, converged = pso.optimize_step(omega, c1, c2)
        
        if step % 10 == 0:
            log_info(f"Step {step}: GBest = {gbest_value:.6f}, Converged = {converged}", "test_pso")
            if metrics:
                log_info(f"  Metrics: {metrics}", "test_pso")
        
        if converged:
            log_info(f"Converged at step {step}", "test_pso")
            break
    
    # Final results
    log_success(f"Final gbest value: {pso.gbest_value:.6f}", "test_pso")
    log_success(f"Final gbest position: {pso.gbest_position[:5]}...", "test_pso")
    log_success("PSO vectorized test completed successfully", "test_pso")

if __name__ == "__main__":
    test_pso_vectorized() 