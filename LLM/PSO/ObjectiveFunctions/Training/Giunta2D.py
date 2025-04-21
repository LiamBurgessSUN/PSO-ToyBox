import numpy as np

from LLM.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class GiuntaFunction(ObjectiveFunction):
    def __init__(self, dim=2, num_particles=30):  # Must be 2D
        assert dim == 2, "Giunta function is only defined for 2D."
        super().__init__(dim, num_particles)
        self.bounds = (-1, 1)

    def evaluate(self, x: np.ndarray) -> float:
        val = 0.6
        for j in range(2):
            a = (16 / 15) * x[j] - 1
            val += np.sin(a) + np.sin(a) ** 2 + (1/50) * np.sin(4 * a)
        return val
