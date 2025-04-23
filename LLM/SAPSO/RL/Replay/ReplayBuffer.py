# File: LLM/RL/Replay/ReplayBuffer.py
# Defines the ReplayBuffer class used for storing and sampling experiences
# in off-policy reinforcement learning algorithms like SAC.

import random # Used for random sampling of experiences
from collections import deque # Efficient data structure for the buffer (double-ended queue)
import numpy as np
import torch

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for off-policy RL agents.

    Stores transitions (state, action, reward, next_state, done) and allows
    sampling of random batches of these transitions for training the agent.
    This helps to break correlations between consecutive experiences and
    improves sample efficiency.
    """
    def __init__(self, capacity, state_dim, action_dim, device="cpu"):
        """
        Initializes the ReplayBuffer.

        Args:
            capacity (int): The maximum number of transitions to store in the buffer.
            state_dim (int): The dimension of the state space. (Used for potential pre-allocation or validation, though not strictly used in this simple implementation).
            action_dim (int): The dimension of the action space. (Used for potential pre-allocation or validation).
            device (str): The device ('cpu' or 'cuda') on which to place the sampled tensors.
        """
        self.capacity = capacity
        # Use deque with maxlen for efficient FIFO buffer implementation
        # Automatically discards the oldest transitions when capacity is reached.
        self.buffer = deque(maxlen=capacity)
        self.state_dim = state_dim # Store state dimension
        self.action_dim = action_dim # Store action dimension
        self.device = device # Store the target device for sampled tensors

    def push(self, state, action, reward, next_state, done):
        """
        Adds a new transition to the buffer.

        Args:
            state (np.ndarray or list): The state observed before taking the action.
            action (np.ndarray or list): The action taken.
            reward (float): The reward received after taking the action.
            next_state (np.ndarray or list): The state observed after taking the action.
            done (bool): A flag indicating whether the episode terminated after this transition.
        """
        # Ensure data is in a consistent format (NumPy arrays) before storing
        # Using float32 for states and actions is standard for neural network inputs.
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        action = np.array(action, dtype=np.float32)
        # Ensure reward is a float
        reward = float(reward)
        # Convert boolean 'done' flag to float (0.0 or 1.0) for easier tensor operations later
        done = float(done)
        # Append the transition tuple to the deque buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Samples a random batch of transitions from the buffer.

        Args:
            batch_size (int): The number of transitions to sample.

        Returns:
            tuple: A tuple containing batches of states, actions, rewards, next_states,
                   and dones as PyTorch tensors on the specified device.
                   (state_batch, action_batch, reward_batch, next_state_batch, done_batch)
        """
        # Randomly select 'batch_size' indices from the buffer
        batch = random.sample(self.buffer, batch_size)

        # Unzip the batch: separates the list of tuples into tuples of states, actions, etc.
        # Example: [(s1,a1,..), (s2,a2,..)] -> ((s1,s2,..), (a1,a2,..), ...)
        # Then, map(np.stack, ...) stacks the elements within each tuple along a new axis (axis 0)
        # creating NumPy arrays for each component (state, action, reward, next_state, done).
        state, action, reward, next_state, done = map(np.stack, zip(*batch))

        # Convert the NumPy arrays to PyTorch tensors
        # Move the tensors to the specified device (e.g., 'cuda' for GPU acceleration)
        state      = torch.tensor(state, dtype=torch.float32).to(self.device)
        action     = torch.tensor(action, dtype=torch.float32).to(self.device)
        # Unsqueeze reward and done to make them column vectors (batch_size, 1)
        # which is often the expected shape for loss calculations.
        reward     = torch.tensor(reward, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        done       = torch.tensor(done, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Return the batch components as tensors
        return state, action, reward, next_state, done

    def __len__(self):
        """
        Returns the current number of transitions stored in the buffer.

        Allows using len(replay_buffer_instance).
        """
        return len(self.buffer)
