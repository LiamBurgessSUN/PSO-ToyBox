import numpy as np

from LLM.PSO.Cognitive.GBest import GlobalBestStrategy
from LLM.PSO.Cognitive.LBest import LocalBestStrategy
from LLM.PSO.Cognitive.PositionSharing import KnowledgeSharingStrategy
from LLM.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction
from LLM.PSO.ObjectiveFunctions.Rastrgin import RastriginFunction
from LLM.PSO.Particle import Particle
from LLM.SwarmMetrics import SwarmMetrics
from LLM.Visualizer import SwarmVisualizer


class PSO:
    def __init__(self, objective_function: ObjectiveFunction,
                 strategy: KnowledgeSharingStrategy,
                 v_clamp_ratio=0.1, use_velocity_clamping=True):
        self.objective_function = objective_function
        self.strategy = strategy
        self.dim = objective_function.dim
        self.bounds = objective_function.bounds
        self.particles = [
            Particle(self.dim, self.bounds)
            for _ in range(objective_function.num_particles)
        ]

        self.gbest_position = self.particles[0].position.copy()
        self.gbest_value = float('inf')

        self.use_velocity_clamping = use_velocity_clamping
        self.v_max = v_clamp_ratio * (self.bounds[1] - self.bounds[0])

        self.metrics_calculator = SwarmMetrics()

    def optimize_step(self, omega, c1, c2):
        for particle in self.particles:
            self._update_velocity(particle, omega, c1, c2)
            self._update_position(particle)
            fitness = self._evaluate(particle)
            if fitness < self.gbest_value:
                self.gbest_value = fitness
                self.gbest_position = particle.position.copy()

        metrics = self.metrics_calculator.compute(self.particles, self.bounds)
        return metrics, self.gbest_value

    def _evaluate(self, particle):
        fitness = self.objective_function.evaluate(particle.position)
        if fitness < particle.pbest_value:
            particle.pbest_value = fitness
            particle.pbest_position = particle.position.copy()
        return fitness

    def _update_velocity(self, particle, omega, c1, c2):
        r1 = np.random.rand(particle.dim)
        r2 = np.random.rand(particle.dim)
        cognitive = c1 * r1 * (particle.pbest_position - particle.position)
        social_target = self.strategy.get_best_position(particle, self.particles)
        social = c2 * r2 * (social_target - particle.position)
        particle.velocity = omega * particle.velocity + cognitive + social

        if self.use_velocity_clamping:
            particle.velocity = np.clip(particle.velocity, -self.v_max, self.v_max)

    def _update_position(self, particle):
        particle.position += particle.velocity
        particle.position = np.clip(particle.position, self.bounds[0], self.bounds[1])

    def _calculate_metrics(self):
        velocities = [np.linalg.norm(p.velocity) for p in self.particles]
        feasible = sum(1 for p in self.particles
                       if np.all(p.position == np.clip(p.position, self.bounds[0], self.bounds[1])))
        stable = sum(1 for p in self.particles if np.linalg.norm(p.velocity) < 1e-3)

        return {
            'avg_velocity': np.mean(velocities),
            'feasible_ratio': feasible / len(self.particles),
            'stable_ratio': stable / len(self.particles)
        }


# # --- Example Usage ---
# if __name__ == "__main__":
#     rastrigin = RastriginFunction()
#
#     # Example: global knowledge (classic gBest)
#     # swarm = PSO(rastrigin, strategy=GlobalBestStrategy(None))
#
#
#     # or use local neighborhood (lBest)
#     swarm = PSO(rastrigin, strategy=LocalBestStrategy(neighborhood_size=2))
#
#     swarm.strategy.swarm = swarm  # Backref needed to access gbest_position
#
#     parameters = [(0.9, 2.0, 0.5), (0.7, 1.5, 1.5), (0.4, 0.5, 2.5)]
#
#     for epoch in range(100):
#         omega, c1, c2 = parameters[min(epoch // 33, 2)]
#         metrics, gbest = swarm.optimize_step(omega, c1, c2)
#         print(f"Epoch {epoch + 1}: Best = {gbest:.4f}, "
#               f"Avg Velocity = {metrics['avg_velocity']:.3f}, "
#               f"Feasible = {metrics['feasible_ratio']:.1%}, "
#               f"Stable = {metrics['stable_ratio']:.1%}")

if __name__ == "__main__":
    rastrigin = RastriginFunction(dim=2, num_particles=20)  # Must be 2D
    rastrigin.plot_3d_surface()
    swarm = PSO(rastrigin, strategy=GlobalBestStrategy(None))
    swarm.strategy.swarm = swarm  # backref

    visualizer = SwarmVisualizer(swarm)

    params = [(0.9, 2.0, 0.5), (0.7, 1.5, 1.5), (0.4, 0.5, 2.5)]
    epoch = [0]

    def pso_step():
        omega, c1, c2 = params[min(epoch[0] // 33, 2)]
        swarm.optimize_step(omega, c1, c2)
        epoch[0] += 1

    visualizer.animate(pso_step, num_steps=100)
