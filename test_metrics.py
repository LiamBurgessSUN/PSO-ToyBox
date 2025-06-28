#!/usr/bin/env python3
"""
Test script for the metrics calculation system.
This script tests the SwarmMetrics functionality.
"""

import numpy as np
from SAPSO_AGENT.SAPSO.PSO.Metrics.Metrics import SwarmMetrics
from SAPSO_AGENT.Logs.logger import log_info, log_header, log_success

def test_metrics():
    """Test the metrics calculation system."""
    log_header("=== Metrics Test ===", "test_metrics")
    
    # Create metrics calculator
    metrics_calc = SwarmMetrics()
    log_info("Created SwarmMetrics instance", "test_metrics")
    
    # Test data
    num_particles = 10
    dim = 5
    
    # Create test positions (within bounds)
    positions = np.random.uniform(-5, 5, (num_particles, dim))
    previous_positions = positions + np.random.uniform(-0.1, 0.1, (num_particles, dim))
    velocities = np.random.uniform(-1, 1, (num_particles, dim))
    bounds = (-5, 5)
    
    # Test parameters
    omega = 0.7
    c1 = 1.5
    c2 = 1.5
    
    log_info(f"Test data: {num_particles} particles, {dim} dimensions", "test_metrics")
    log_info(f"PSO parameters: ω={omega}, c1={c1}, c2={c2}", "test_metrics")
    
    # Calculate metrics
    metrics = metrics_calc.compute(
        positions=positions,
        previous_positions=previous_positions,
        velocities=velocities,
        bounds=bounds,
        omega=omega,
        c1=c1,
        c2=c2
    )
    
    # Display results
    log_info("Calculated metrics:", "test_metrics")
    for key, value in metrics.items():
        if isinstance(value, float):
            log_info(f"  {key}: {value:.6f}", "test_metrics")
        else:
            log_info(f"  {key}: {value}", "test_metrics")
    
    # Test stability condition
    is_stable = metrics_calc._check_poli_stability(omega, c1, c2)
    log_info(f"Poli's stability condition: {is_stable}", "test_metrics")
    
    # Test with some particles out of bounds
    log_info("Testing with infeasible particles...", "test_metrics")
    positions_out = positions.copy()
    positions_out[0] = np.array([10, 10, 10, 10, 10])  # Out of bounds
    
    metrics_out = metrics_calc.compute(
        positions=positions_out,
        previous_positions=previous_positions,
        velocities=velocities,
        bounds=bounds,
        omega=omega,
        c1=c1,
        c2=c2
    )
    
    log_info(f"Infeasible ratio: {metrics_out['infeasible_ratio']:.3f}", "test_metrics")
    
    # Test edge cases
    log_info("Testing edge cases...", "test_metrics")
    
    # Single particle
    single_pos = np.random.uniform(-5, 5, (1, dim))
    single_prev = single_pos + np.random.uniform(-0.1, 0.1, (1, dim))
    single_vel = np.random.uniform(-1, 1, (1, dim))
    
    single_metrics = metrics_calc.compute(
        positions=single_pos,
        previous_positions=single_prev,
        velocities=single_vel,
        bounds=bounds,
        omega=omega,
        c1=c1,
        c2=c2
    )
    
    log_info(f"Single particle diversity: {single_metrics['swarm_diversity']:.6f}", "test_metrics")
    
    log_success("Metrics test completed successfully", "test_metrics")

if __name__ == "__main__":
    test_metrics() 