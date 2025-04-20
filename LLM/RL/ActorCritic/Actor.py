# --- Imports remain the same ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

LOG_STD_MIN = -20
LOG_STD_MAX = 2

class Actor(nn.Module):
    # --- ADDED adaptive_nt flag ---
    def __init__(self, state_dim, action_dim, hidden_dim=256, adaptive_nt=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        # --- MODIFIED output layer size determination ---
        self.output_dim = action_dim # Should be 3 or 4 based on adaptive_nt
        # self.adaptive_nt = adaptive_nt # Store if needed elsewhere

        # Output layers for mean and log_std
        self.mean = nn.Linear(hidden_dim, self.output_dim)
        self.log_std = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z) # Action components are in [-1, 1]

        # Log prob correction for tanh squashing
        # Ensure sum is over the last dimension (action dimension)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def get_deterministic_action(self, state):
        x = self.net(state)
        mean = self.mean(x)
        return torch.tanh(mean)