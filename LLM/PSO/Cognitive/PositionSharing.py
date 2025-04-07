from abc import abstractmethod, ABC

import numpy as np


class KnowledgeSharingStrategy(ABC):
    @abstractmethod
    def get_best_position(self, particle, swarm_particles) -> np.ndarray:
        """Return the position the particle should be attracted to."""
        pass