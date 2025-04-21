# File: LLM/RL/ActorCritic/Agent.py
# Refactored to use the logger module with optional debug logging.
# Removed import checks for Actor/Critic.

import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import os # Needed for path checks in load
import traceback # For logging exceptions
from pathlib import Path # To get module name

# --- Import Logger ---
# Using the specified import path: from LLM.Logs import logger
try:
    # Import the module first if needed, then specific functions
    from LLM.Logs import logger # Assuming standard structure
    from LLM.Logs.logger import log_info, log_error, log_warning, log_success, log_header, log_debug
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

# --- Project Imports ---
# Assuming Actor and Critic are in the same directory
# Removed try-except block as requested
from .Actor import Actor
from .Critic import QNetwork


# --- Module Name for Logging ---
module_name = Path(__file__).stem # Gets 'Agent'

class SACAgent:
    """
    Soft Actor-Critic Agent implementation.

    Handles network initialization, action selection, training updates,
    and model saving/loading. Includes optional debug logging.
    """
    def __init__(
        self,
        state_dim,
        action_dim, # This should be 3 if fixed nt, 4 if adaptive
        hidden_dim=256,
        gamma=0.99, # Note: Paper uses gamma=1.0
        tau=0.005,
        alpha=0.2,
        actor_lr=3e-4,
        critic_lr=3e-4,
        device="cpu",
        adaptive_nt=False # Pass the flag
    ):
        """
        Initializes the SAC Agent.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space (3 or 4).
            hidden_dim (int): Size of the hidden layers in networks.
            gamma (float): Discount factor.
            tau (float): Target network soft update coefficient.
            alpha (float): Entropy regularization coefficient.
            actor_lr (float): Learning rate for the actor network.
            critic_lr (float): Learning rate for the critic networks.
            device (str): Device to run tensors on ('cpu' or 'cuda').
            adaptive_nt (bool): Whether the agent controls the 'nt' parameter.
        """
        log_info(f"Initializing SACAgent on device: {device}", module_name)
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.adaptive_nt = adaptive_nt # Store flag
        self.action_dim = action_dim # Store correct action dim (3 or 4)

        # --- Initialize Networks ---
        try:
            # Pass adaptive_nt and correct action_dim to Actor
            self.actor = Actor(state_dim, self.action_dim, hidden_dim, adaptive_nt=self.adaptive_nt).to(device)
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

            # Critic networks need correct action_dim
            self.q1 = QNetwork(state_dim, self.action_dim, hidden_dim).to(device)
            self.q2 = QNetwork(state_dim, self.action_dim, hidden_dim).to(device)
            # Target networks are crucial for stability
            self.q1_target = deepcopy(self.q1).eval().to(device)
            self.q2_target = deepcopy(self.q2).eval().to(device)

            self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=critic_lr)
            self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=critic_lr)

            self.loss_fn = nn.MSELoss()
            log_info("SAC networks and optimizers initialized.", module_name)
        except Exception as e:
            log_error(f"Error initializing agent networks/optimizers: {e}", module_name)
            log_error(traceback.format_exc(), module_name)
            raise # Re-raise the exception to prevent using an uninitialized agent


    def select_action(self, state, deterministic=False):
        """
        Selects an action based on the current state.

        Args:
            state (np.ndarray): The current state observation.
            deterministic (bool): If True, returns the mean action (for evaluation).
                                  If False, samples from the action distribution (for training).

        Returns:
            np.ndarray: The selected action.
        """
        # Ensure state is a PyTorch tensor on the correct device
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad(): # Inference does not require gradient calculation
            if deterministic:
                action = self.actor.get_deterministic_action(state)
            else:
                action, _ = self.actor(state) # Get sampled action and log probability
        # Detach from graph, move to CPU, convert to NumPy array, remove batch dim
        return action.detach().cpu().numpy()[0]


    def update(self, replay_buffer, batch_size):
        """
        Performs a single training update step for the agent.

        Samples a batch from the replay buffer and updates the Actor and Critic networks.

        Args:
            replay_buffer (ReplayBuffer): The experience replay buffer.
            batch_size (int): The number of experiences to sample for the update.
        """
        # Sample a batch of experiences
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        # Tensors are already on self.device from the buffer's sample method

        # --- Critic Update ---
        with torch.no_grad(): # Target calculations don't need gradients
            # Get next action and its log probability from the current actor policy
            next_action, next_logp = self.actor(next_state)

            # Calculate target Q-values using target critic networks
            target_q1 = self.q1_target(next_state, next_action)
            target_q2 = self.q2_target(next_state, next_action)
            # Use the minimum of the two target Q-values (clipped double-Q learning)
            target_q_min = torch.min(target_q1, target_q2)

            # Add entropy term to the target Q-value
            target_q = target_q_min - self.alpha * next_logp

            # Calculate the final target value (Bellman equation)
            target = reward + (1.0 - done) * self.gamma * target_q

        # Calculate current Q-values using current critic networks
        current_q1 = self.q1(state, action)
        current_q2 = self.q2(state, action)

        # Calculate critic losses (MSE between current and target Q-values)
        q1_loss = self.loss_fn(current_q1, target)
        q2_loss = self.loss_fn(current_q2, target)

        # Optimize Q1 network
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        # Optimize Q2 network
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # --- Actor Update ---
        # Freeze Q-networks to avoid computing gradients for them during actor update
        for param in self.q1.parameters(): param.requires_grad = False
        for param in self.q2.parameters(): param.requires_grad = False

        # Get new action and log probability for the current state from the actor
        new_action, log_prob = self.actor(state)

        # Calculate Q-values for the new action using the (now frozen) critic networks
        q1_val = self.q1(state, new_action)
        q2_val = self.q2(state, new_action)
        q_val_min = torch.min(q1_val, q2_val)

        # Calculate actor loss: aim to maximize Q-value and entropy
        actor_loss = (self.alpha * log_prob - q_val_min).mean()

        # Optimize Actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze Q-networks
        for param in self.q1.parameters(): param.requires_grad = True
        for param in self.q2.parameters(): param.requires_grad = True

        # --- Optional Debug Logging ---
        # log_debug(f"Losses -> Q1: {q1_loss.item():.4f}, Q2: {q2_loss.item():.4f}, Actor: {actor_loss.item():.4f}", module_name)
        # log_debug(f"Mean Q-target: {target.mean().item():.4f}, Mean log_prob: {log_prob.mean().item():.4f}", module_name)
        # --- End Optional Debug ---


        # --- Soft Update Target Networks ---
        # Gradually update target networks towards the main networks
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)


    def _soft_update(self, source_net, target_net):
        """Performs a soft update of the target network parameters."""
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path):
        """Saves the agent's networks and optimizer states."""
        log_info(f"Attempting to save agent checkpoint to: {path}", module_name)
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({
                'actor_state_dict': self.actor.state_dict(),
                'q1_state_dict': self.q1.state_dict(),
                'q2_state_dict': self.q2.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'q1_optimizer_state_dict': self.q1_optimizer.state_dict(),
                'q2_optimizer_state_dict': self.q2_optimizer.state_dict(),
                # Optionally save other hyperparameters if needed for reloading
                'adaptive_nt': self.adaptive_nt,
                'action_dim': self.action_dim
            }, path)
            log_success(f"Agent checkpoint successfully saved to {path}", module_name)
        except Exception as e:
            log_error(f"Failed to save agent checkpoint to {path}: {e}", module_name)
            log_error(traceback.format_exc(), module_name)


    def load(self, path):
        """Loads the agent's networks and optimizer states from a checkpoint file."""
        log_info(f"Attempting to load agent checkpoint from: {path}", module_name)
        if not os.path.exists(path):
            log_error(f"Checkpoint file not found at {path}. Cannot load agent.", module_name)
            return # Or raise FileNotFoundError

        try:
            checkpoint = torch.load(path, map_location=self.device) # Load to the agent's device

            # Load network states
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.q1.load_state_dict(checkpoint['q1_state_dict'])
            self.q2.load_state_dict(checkpoint['q2_state_dict'])

            # Load optimizer states
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer_state_dict'])
            self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer_state_dict'])

            # --- Important: Re-initialize target networks after loading main networks ---
            self.q1_target = deepcopy(self.q1).eval().to(self.device)
            self.q2_target = deepcopy(self.q2).eval().to(self.device)

            # Optionally load and verify other saved parameters
            loaded_adaptive_nt = checkpoint.get('adaptive_nt', None)
            loaded_action_dim = checkpoint.get('action_dim', None)
            if loaded_adaptive_nt is not None and loaded_adaptive_nt != self.adaptive_nt:
                log_warning(f"Loaded checkpoint adaptive_nt ({loaded_adaptive_nt}) differs from agent config ({self.adaptive_nt}).", module_name)
            if loaded_action_dim is not None and loaded_action_dim != self.action_dim:
                 log_warning(f"Loaded checkpoint action_dim ({loaded_action_dim}) differs from agent config ({self.action_dim}).", module_name)


            log_success(f"Agent checkpoint successfully loaded from {path}", module_name)

        except KeyError as e:
            log_error(f"Failed to load checkpoint from {path}. Missing key: {e}", module_name)
            log_error("Ensure the checkpoint file is valid and contains all required keys.", module_name)
            log_error(traceback.format_exc(), module_name)
        except Exception as e:
            log_error(f"Failed to load agent checkpoint from {path}: {e}", module_name)
            log_error(traceback.format_exc(), module_name)

