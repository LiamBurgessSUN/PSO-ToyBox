import numpy as np

from LLM.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class EggHolderFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-512, 512)

    def evaluate(self, x: np.ndarray) -> float:
        return np.sum(
            -(x[1:] + 47) * np.sin(np.sqrt(np.abs(x[1:] + x[:-1] / 2 + 47))) -
            x[:-1] * np.sin(np.sqrt(np.abs(x[:-1] - (x[1:] + 47))))
        )
