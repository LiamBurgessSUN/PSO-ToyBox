#!/usr/bin/env python3
"""
Debug script for testing the SAC agent.
This script allows you to debug the agent without running the full training pipeline.
"""

import torch
import numpy as np
from SAPSO_AGENT.SAPSO.RL.ActorCritic.Agent import SACAgent
from SAPSO_AGENT.SAPSO.RL.Replay.ReplayBuffer import ReplayBuffer
from SAPSO_AGENT.Logs.logger import log_info, log_header

def debug_agent():
    """Debug the SAC agent step by step."""
    log_header("=== SAC Agent Debug ===", "debug_agent")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_info(f"Using device: {device}", "debug_agent")
    
    # Agent parameters
    state_dim = 4  # Environment observation space
    action_dim = 3  # Fixed Nt mode
    hidden_dim = 64  # Smaller for debugging
    
    # Create agent
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        actor_lr=3e-4,
        critic_lr=3e-4,
        device=device,
        adaptive_nt=False
    )
    log_info("SAC agent created successfully", "debug_agent")
    
    # Create replay buffer
    buffer = ReplayBuffer(
        capacity=1000,
        state_dim=state_dim,
        action_dim=action_dim,
        device=device
    )
    log_info("Replay buffer created successfully", "debug_agent")
    
    # Test action selection
    test_state = np.random.randn(state_dim).astype(np.float32)
    log_info(f"Test state: {test_state}", "debug_agent")
    
    # Deterministic action
    det_action = agent.select_action(test_state, deterministic=True)
    log_info(f"Deterministic action: {det_action}", "debug_agent")
    
    # Stochastic action
    stoch_action = agent.select_action(test_state, deterministic=False)
    log_info(f"Stochastic action: {stoch_action}", "debug_agent")
    
    # Test replay buffer
    for i in range(10):
        state = np.random.randn(state_dim).astype(np.float32)
        action = np.random.randn(action_dim).astype(np.float32)
        reward = np.random.randn().astype(np.float32)
        next_state = np.random.randn(state_dim).astype(np.float32)
        done = np.random.choice([True, False]).astype(np.float32)
        
        buffer.add(state, action, reward, next_state, done)
    
    log_info(f"Added 10 transitions to replay buffer. Buffer size: {len(buffer)}", "debug_agent")
    
    # Test sampling
    if len(buffer) >= 5:
        batch = buffer.sample(5)
        log_info(f"Sampled batch shapes: {[t.shape for t in batch]}", "debug_agent")
        
        # Test agent update
        try:
            agent.update(buffer, batch_size=5)
            log_info("Agent update completed successfully", "debug_agent")
        except Exception as e:
            log_info(f"Agent update failed: {e}", "debug_agent")
    
    log_info("SAC agent debug completed", "debug_agent")

if __name__ == "__main__":
    debug_agent() 