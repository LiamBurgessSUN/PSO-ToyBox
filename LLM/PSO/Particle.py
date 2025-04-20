import numpy as np


class Particle:
    def __init__(self, dim, bounds):
        self.dim = dim
        self.bounds = bounds

        self.position = np.random.uniform(low=bounds[0], high=bounds[1], size=dim)
        self.velocity = np.zeros(dim)

        self.pbest_position = self.position.copy()
        self.pbest_value = float('inf')

        self.is_active = True  # Add this attribute
        self.stagnation_counter = 0  # Optional: for stagnation condition
        self.velocity_threshold_counter = 0  # Optional: for velocity condition