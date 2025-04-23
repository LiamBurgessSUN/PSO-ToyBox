# File: LLM/RL/ActorCritic/Actor.py
# Defines the Actor network for the Soft Actor-Critic (SAC) algorithm.
# The Actor represents the policy function (pi_phi), mapping states to a
# probability distribution over actions.

import torch
import torch.nn as nn
import torch.distributions as distributions # For creating probability distributions

# Define numerical stability constants for log standard deviation
LOG_STD_MIN = -20 # Minimum value for log standard deviation
LOG_STD_MAX = 2   # Maximum value for log standard deviation

class Actor(nn.Module):
    """
    The Actor network (policy network) for the SAC algorithm.

    It takes the environment state as input and outputs a probability distribution
    over the action space. For continuous action spaces like in SAC-SAPSO,
    this is typically a Gaussian distribution parameterized by its mean and
    standard deviation. The network learns the optimal policy by adjusting
    its parameters based on the critic's evaluation and the entropy objective.
    """
    # --- ADDED adaptive_nt flag ---
    def __init__(self, state_dim, action_dim, hidden_dim=256, adaptive_nt=False):
        """
        Initializes the Actor network.

        Args:
            state_dim (int): Dimension of the input state space.
            action_dim (int): Dimension of the output action space.
                              (3 if only adapting omega, c1, c2; 4 if also adapting nt).
            hidden_dim (int): Number of neurons in the hidden layers.
            adaptive_nt (bool): Flag indicating if the agent adapts the 'nt' parameter.
                                This determines the output dimension of the network.
        """
        super().__init__() # Call the constructor of the parent class (nn.Module)
        self.adaptive_nt = adaptive_nt # Store the adaptive_nt flag

        # Define the main network body (MLP)
        # Takes state_dim as input, passes through hidden layers with ReLU activation
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            # The final hidden layer output will be fed into the mean and log_std layers
        )
        self.output_dim = action_dim # Store the correct action dimension (3 or 4)

        # Define the output layers:
        # One layer outputs the mean of the Gaussian distribution for each action dimension.
        self.mean = nn.Linear(hidden_dim, self.output_dim)
        # Another layer outputs the logarithm of the standard deviation (log_std) for stability.
        self.log_std = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, state):
        """
        Performs the forward pass of the Actor network.

        Takes a state tensor and outputs a sampled action and its log probability.

        Args:
            state (torch.Tensor): The input state tensor (batch_size, state_dim).

        Returns:
            tuple:
                - action (torch.Tensor): The action sampled from the policy distribution,
                                         squashed to the range [-1, 1] using tanh.
                                         Shape: (batch_size, action_dim).
                - log_prob (torch.Tensor): The log probability of the sampled action *before*
                                           squashing, adjusted for the tanh transformation.
                                           Shape: (batch_size, 1).
        """
        # Pass the state through the main network body
        x = self.net(state)

        # Calculate the mean of the action distribution
        mean = self.mean(x)

        # Calculate the log standard deviation, clamping it for numerical stability
        log_std = self.log_std(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        # Calculate the standard deviation by exponentiating the log_std
        std = log_std.exp()

        # Create a Normal (Gaussian) distribution using the calculated mean and std
        # This represents the policy distribution before squashing
        normal = distributions.Normal(mean, std)

        # Sample an action 'z' from the Normal distribution using the reparameterization trick (rsample)
        # rsample allows gradients to flow back through the sampling process
        z = normal.rsample()

        # Squash the sampled action 'z' into the range [-1, 1] using the hyperbolic tangent (tanh) function
        # This ensures the output actions are bounded, which is common practice in SAC.
        # These squashed actions will be used by the agent to interact with the environment.
        action = torch.tanh(z)

        # Calculate the log probability of the *unsquashed* action 'z' under the Normal distribution
        # Then, apply the correction term for the tanh transformation.
        # This correction is crucial for correctly optimizing the policy under the squashed action space.
        # log pi(a|s) = log Normal(z|s) - log(1 - tanh(z)^2)
        # Add a small epsilon (1e-6) for numerical stability to avoid log(0).
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)

        # Sum the log probabilities across the action dimensions.
        # The policy aims to maximize the expected log probability (entropy).
        log_prob = log_prob.sum(dim=-1, keepdim=True) # Keep dimension for broadcasting later

        return action, log_prob

    def get_deterministic_action(self, state):
        """
        Gets the deterministic action for a given state (used during evaluation).

        Instead of sampling, it returns the mean of the action distribution, squashed by tanh.

        Args:
            state (torch.Tensor): The input state tensor (batch_size, state_dim).

        Returns:
            torch.Tensor: The deterministic action, squashed to [-1, 1].
                          Shape: (batch_size, action_dim).
        """
        # Pass the state through the main network body
        x = self.net(state)
        # Calculate the mean of the action distribution
        mean = self.mean(x)
        # Return the squashed mean action (tanh applied to the mean)
        return torch.tanh(mean)

