import numpy as np


class SwarmMetrics:
    def __init__(self):
        self._metric_functions = []

        # Register default metrics
        self.register("avg_velocity", self._average_velocity)
        self.register("feasible_ratio", self._feasibility_ratio)
        self.register("stable_ratio", self._stability_ratio)

    def register(self, name: str, func):
        """Register a new metric function by name."""
        self._metric_functions.append((name, func))

    def compute(self, swarm_particles, bounds):
        """Compute all registered metrics and return as a dict."""
        return {
            name: func(swarm_particles, bounds)
            for name, func in self._metric_functions
        }

    @staticmethod
    def _average_velocity(particles, _):
        velocities = [np.linalg.norm(p.velocity) for p in particles]
        return np.mean(velocities)

    @staticmethod
    def _feasibility_ratio(particles, bounds):
        feasible = sum(1 for p in particles
                       if np.all(p.position == np.clip(p.position, bounds[0], bounds[1])))
        return feasible / len(particles)

    @staticmethod
    def _stability_ratio(particles, _):
        stable = sum(1 for p in particles if np.linalg.norm(p.velocity) < 1e-3)
        return stable / len(particles)
