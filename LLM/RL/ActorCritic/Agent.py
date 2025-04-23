# File: LLM/RL/ActorCritic/Agent.py
# Implements the Soft Actor-Critic (SAC) agent, which orchestrates the
# interaction between the Actor, Critic, and the environment (PSO simulation).
# It handles learning updates, action selection, and model persistence.

import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy # Used to create target networks
import os # Needed for path checks in load/save
import traceback # For logging exceptions
from pathlib import Path # To get module name

from LLM.Logs.logger import log_info, log_error, log_warning, log_success, log_header, log_debug

# --- Project Imports ---
# Assuming Actor and Critic are in the same directory (ActorCritic)
# Direct relative import is standard practice within a package.
from Actor import Actor
from Critic import QNetwork


# --- Module Name for Logging ---
module_name = Path(__file__).stem # Gets 'Agent'

class SACAgent:
    """
    Soft Actor-Critic Agent implementation.

    Handles network initialization, action selection, training updates using
    experiences from a replay buffer, and saving/loading the trained model.
    Implements key SAC features like entropy maximization, clipped double-Q
    learning, and target network updates.
    """
    def __init__(
        self,
        state_dim,
        action_dim, # This should be 3 if fixed nt, 4 if adaptive
        hidden_dim=256,
        gamma=0.99, # Discount factor (Note: Paper uses gamma=1.0 for PSO env)
        tau=0.005,  # Target network soft update coefficient (Polyak averaging)
        alpha=0.2,  # Entropy regularization coefficient (temperature parameter)
        actor_lr=3e-4, # Learning rate for the actor network
        critic_lr=3e-4,# Learning rate for the critic networks
        device="cpu", # Device to run computations on ('cpu' or 'cuda')
        adaptive_nt=False # Flag to indicate if 'nt' is part of the action space
    ):
        """
        Initializes the SAC Agent.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space (3 or 4).
            hidden_dim (int): Size of the hidden layers in networks.
            gamma (float): Discount factor for future rewards.
            tau (float): Coefficient for soft target network updates.
            alpha (float): Weighting factor for the entropy term in the objective.
            actor_lr (float): Learning rate for the actor optimizer.
            critic_lr (float): Learning rate for the critic optimizers.
            device (str): Computation device ('cpu' or 'cuda').
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
            # Actor Network (Policy)
            # Pass adaptive_nt and correct action_dim to Actor constructor
            self.actor = Actor(state_dim, self.action_dim, hidden_dim, adaptive_nt=self.adaptive_nt).to(device)
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

            # Critic Networks (Q-Value Functions)
            # SAC uses two Q-networks to reduce overestimation bias.
            self.q1 = QNetwork(state_dim, self.action_dim, hidden_dim).to(device)
            self.q2 = QNetwork(state_dim, self.action_dim, hidden_dim).to(device)

            # Target Networks for Critics
            # These are slowly updated copies of the main Q-networks, used for stable target calculation.
            # Use deepcopy to ensure they are independent initially. .eval() sets them to evaluation mode.
            self.q1_target = deepcopy(self.q1).eval().to(device)
            self.q2_target = deepcopy(self.q2).eval().to(device)

            # Optimizers for the Q-networks
            self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=critic_lr)
            self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=critic_lr)

            # Loss function for Q-network updates (typically Mean Squared Error)
            self.loss_fn = nn.MSELoss()
            log_info("SAC networks and optimizers initialized.", module_name)
        except Exception as e:
            log_error(f"Error initializing agent networks/optimizers: {e}", module_name)
            log_error(traceback.format_exc(), module_name)
            raise # Re-raise the exception to prevent using an uninitialized agent


    def select_action(self, state, deterministic=False):
        """
        Selects an action based on the current state using the actor network.

        Args:
            state (np.ndarray): The current state observation from the environment.
            deterministic (bool): If True, returns the mean action (for evaluation/testing).
                                  If False, samples stochastically from the action
                                  distribution (for training/exploration).

        Returns:
            np.ndarray: The selected action, scaled to the environment's action range
                        (initially in [-1, 1] from tanh).
        """
        # Convert state (numpy array) to a PyTorch tensor, add batch dimension, move to device
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad(): # Disable gradient calculations for inference
            if deterministic:
                # Get the deterministic action (mean of the distribution)
                action = self.actor.get_deterministic_action(state)
            else:
                # Sample an action stochastically from the policy distribution
                action, _ = self.actor(state) # We only need the action here, not the log_prob
        # Detach the action tensor from the computation graph, move to CPU, convert to NumPy array,
        # and remove the batch dimension before returning.
        return action.detach().cpu().numpy()[0]


    def update(self, replay_buffer, batch_size):
        """
        Performs a single training update step for the SAC agent.

        This involves sampling a batch from the replay buffer, calculating target
        values, updating the critic (Q) networks, updating the actor (policy)
        network, and softly updating the target networks.

        Args:
            replay_buffer (ReplayBuffer): The experience replay buffer to sample from.
            batch_size (int): The number of transitions to sample for the update.
        """
        # 1. Sample a batch of transitions from the replay buffer
        # Tensors are returned on the agent's device (specified during buffer init)
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        # --- 2. Critic Network Update ---
        with torch.no_grad(): # Target calculations should not affect gradients
            # a. Get the next action and its log probability from the *current* actor policy
            #    This is needed for the entropy term in the target Q calculation.
            next_action, next_logp = self.actor(next_state)

            # b. Calculate target Q-values using the *target* critic networks
            #    This provides a stable target for the main critic networks to learn towards.
            target_q1 = self.q1_target(next_state, next_action)
            target_q2 = self.q2_target(next_state, next_action)

            # c. Clipped Double-Q Learning: Use the minimum of the two target Q-values
            #    This helps prevent overestimation of Q-values.
            target_q_min = torch.min(target_q1, target_q2)

            # d. Incorporate entropy: Subtract the scaled log probability (alpha * next_logp)
            #    This encourages the policy to maintain high entropy (explore more).
            #    This is the "soft" part of Soft Actor-Critic.
            target_q = target_q_min - self.alpha * next_logp

            # e. Calculate the final target value using the Bellman equation:
            #    target = r + gamma * (1 - done) * E[Q_target(s', a') - alpha * log pi(a'|s')]
            #    The (1 - done) term ensures that the value of terminal states is just the reward.
            target = reward + (1.0 - done) * self.gamma * target_q

        # f. Calculate current Q-values using the main critic networks for the sampled state-action pairs
        current_q1 = self.q1(state, action)
        current_q2 = self.q2(state, action)

        # g. Calculate critic losses: Mean Squared Error between current Q-values and the calculated target
        q1_loss = self.loss_fn(current_q1, target)
        q2_loss = self.loss_fn(current_q2, target)

        # h. Optimize Q1 network: Zero gradients, compute gradients, update weights
        self.q1_optimizer.zero_grad()
        q1_loss.backward() # Compute gradients for Q1
        self.q1_optimizer.step() # Update Q1 weights

        # i. Optimize Q2 network: Zero gradients, compute gradients, update weights
        self.q2_optimizer.zero_grad()
        q2_loss.backward() # Compute gradients for Q2
        self.q2_optimizer.step() # Update Q2 weights

        # --- 3. Actor Network Update ---
        # Freeze Q-networks' parameters to avoid computing their gradients during the actor update.
        # We only want gradients for the actor network based on the Q-values.
        for param in self.q1.parameters(): param.requires_grad = False
        for param in self.q2.parameters(): param.requires_grad = False

        # a. Get new actions and their log probabilities for the *current* states from the actor policy
        new_action, log_prob = self.actor(state)

        # b. Calculate Q-values for these new state-action pairs using the (frozen) main critic networks
        q1_val = self.q1(state, new_action)
        q2_val = self.q2(state, new_action)
        # Use the minimum Q-value for the actor update (consistent with clipped double-Q)
        q_val_min = torch.min(q1_val, q2_val)

        # c. Calculate actor loss: Maximize E[Q(s,a) - alpha * log pi(a|s)]
        #    This is equivalent to minimizing E[alpha * log pi(a|s) - Q(s,a)].
        #    The loss encourages the actor to take actions that lead to higher Q-values
        #    while also maintaining high entropy (high log_prob).
        actor_loss = (self.alpha * log_prob - q_val_min).mean() # Average loss over the batch

        # d. Optimize Actor network: Zero gradients, compute gradients, update weights
        self.actor_optimizer.zero_grad()
        actor_loss.backward() # Compute gradients for the actor
        self.actor_optimizer.step() # Update actor weights

        # e. Unfreeze Q-networks' parameters for the next update cycle
        for param in self.q1.parameters(): param.requires_grad = True
        for param in self.q2.parameters(): param.requires_grad = True

        # --- Optional Debug Logging ---
        # log_debug(f"Losses -> Q1: {q1_loss.item():.4f}, Q2: {q2_loss.item():.4f}, Actor: {actor_loss.item():.4f}", module_name)
        # log_debug(f"Mean Q-target: {target.mean().item():.4f}, Mean log_prob: {log_prob.mean().item():.4f}", module_name)
        # --- End Optional Debug ---

        # --- 4. Soft Update Target Networks ---
        # Gradually update the target networks towards the main networks using Polyak averaging.
        # This provides stability compared to directly copying weights.
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)


    def _soft_update(self, source_net, target_net):
        """
        Performs a soft update (Polyak averaging) of the target network parameters.
        target_weights = tau * source_weights + (1 - tau) * target_weights

        Args:
            source_net (nn.Module): The main network (actor or critic).
            target_net (nn.Module): The corresponding target network.
        """
        # Iterate over parameters of both networks simultaneously
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            # Update the target network's parameter data in-place
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path):
        """
        Saves the agent's state (network weights and optimizer states) to a file.

        Args:
            path (str): The file path where the checkpoint will be saved.
        """
        log_info(f"Attempting to save agent checkpoint to: {path}", module_name)
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Save the state dictionaries of networks and optimizers
            torch.save({
                'actor_state_dict': self.actor.state_dict(),
                'q1_state_dict': self.q1.state_dict(),
                'q2_state_dict': self.q2.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'q1_optimizer_state_dict': self.q1_optimizer.state_dict(),
                'q2_optimizer_state_dict': self.q2_optimizer.state_dict(),
                # Save relevant configuration flags used during initialization
                'adaptive_nt': self.adaptive_nt,
                'action_dim': self.action_dim
            }, path)
            log_success(f"Agent checkpoint successfully saved to {path}", module_name)
        except Exception as e:
            log_error(f"Failed to save agent checkpoint to {path}: {e}", module_name)
            log_error(traceback.format_exc(), module_name)


    def load(self, path):
        """
        Loads the agent's state from a saved checkpoint file.

        Args:
            path (str): The file path of the checkpoint to load.
        """
        log_info(f"Attempting to load agent checkpoint from: {path}", module_name)
        if not os.path.exists(path):
            log_error(f"Checkpoint file not found at {path}. Cannot load agent.", module_name)
            # Optionally raise FileNotFoundError(f"Checkpoint file not found: {path}")
            return # Exit loading if file doesn't exist

        try:
            # Load the checkpoint dictionary, ensuring it's loaded onto the agent's current device
            checkpoint = torch.load(path, map_location=self.device)

            # Load network state dictionaries
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.q1.load_state_dict(checkpoint['q1_state_dict'])
            self.q2.load_state_dict(checkpoint['q2_state_dict'])

            # Load optimizer state dictionaries
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer_state_dict'])
            self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer_state_dict'])

            # --- Important: Re-initialize target networks ---
            # After loading the main networks, the target networks must be reset
            # to be exact copies of the loaded main networks initially.
            self.q1_target = deepcopy(self.q1).eval().to(self.device)
            self.q2_target = deepcopy(self.q2).eval().to(self.device)

            # Optionally load and verify configuration flags saved in the checkpoint
            loaded_adaptive_nt = checkpoint.get('adaptive_nt', None) # Use .get for safety
            loaded_action_dim = checkpoint.get('action_dim', None)
            if loaded_adaptive_nt is not None and loaded_adaptive_nt != self.adaptive_nt:
                log_warning(f"Loaded checkpoint adaptive_nt ({loaded_adaptive_nt}) differs from agent config ({self.adaptive_nt}). Ensure this is intended.", module_name)
            if loaded_action_dim is not None and loaded_action_dim != self.action_dim:
                 log_warning(f"Loaded checkpoint action_dim ({loaded_action_dim}) differs from agent config ({self.action_dim}). Ensure this is intended.", module_name)
                 # Could potentially update self.action_dim here if needed, but warning is safer

            log_success(f"Agent checkpoint successfully loaded from {path}", module_name)

        except KeyError as e:
            # Handle cases where the checkpoint file is missing expected keys
            log_error(f"Failed to load checkpoint from {path}. Missing key: {e}", module_name)
            log_error("Ensure the checkpoint file is valid and contains all required keys (actor_state_dict, q1_state_dict, etc.).", module_name)
            log_error(traceback.format_exc(), module_name)
        except Exception as e:
            # Catch any other potential errors during loading
            log_error(f"Failed to load agent checkpoint from {path}: {e}", module_name)
            log_error(traceback.format_exc(), module_name)

