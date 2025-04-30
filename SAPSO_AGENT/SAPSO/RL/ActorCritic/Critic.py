# File: SAPSO_AGENT/RL/ActorCritic/Critic.py
# Defines the Critic network (Q-Network) for the Soft Actor-Critic (SAC) algorithm.
# The Critic estimates the expected cumulative discounted reward (Q-value)
# for taking a specific action in a given state.

import torch
import torch.nn as nn

class QNetwork(nn.Module):
    """
    The Critic network (Q-Network) used in the SAC algorithm.

    It approximates the action-value function Q(s, a), which represents the
    expected return (cumulative discounted reward) starting from state 's',
    taking action 'a', and following the current policy thereafter.
    SAC typically uses two independent Q-networks (and their targets) to
    mitigate overestimation bias (Clipped Double-Q Learning).
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        Initializes the Q-Network.

        Args:
            state_dim (int): Dimension of the input state space.
            action_dim (int): Dimension of the input action space.
            hidden_dim (int): Number of neurons in the hidden layers.
        """
        super().__init__() # Call the constructor of the parent class (nn.Module)

        # Define the network architecture using nn.Sequential
        # It takes the concatenated state and action as input.
        self.net = nn.Sequential(
            # First linear layer takes state_dim + action_dim inputs
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(), # ReLU activation function
            # Second hidden layer
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), # ReLU activation function
            # Output layer: predicts a single Q-value
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        """
        Performs the forward pass of the Q-Network.

        Takes state and action tensors and outputs the estimated Q-value.

        Args:
            state (torch.Tensor): The input state tensor (batch_size, state_dim).
            action (torch.Tensor): The input action tensor (batch_size, action_dim).

        Returns:
            torch.Tensor: The estimated Q-value for each state-action pair in the batch.
                          Shape: (batch_size, 1).
        """
        # Concatenate the state and action tensors along the last dimension
        # This combined tensor serves as the input to the network.
        x = torch.cat([state, action], dim=-1)

        # Pass the concatenated input through the network layers
        q_value = self.net(x)

        # Return the predicted Q-value
        return q_value
