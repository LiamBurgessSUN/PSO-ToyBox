from SAPSO_AGENT.SAPSO.PSO.Cognitive.PositionSharing import KnowledgeSharingStrategy


class LocalBestStrategy(KnowledgeSharingStrategy):
    def __init__(self, neighborhood_size=2):
        self.neighborhood_size = neighborhood_size

    def get_best_position(self, particle, swarm_particles):
        index = swarm_particles.index(particle)
        total = len(swarm_particles)

        # Get neighbor indices (wrap-around ring topology)
        neighbors = [(index + i) % total for i in range(-self.neighborhood_size, self.neighborhood_size + 1)]
        best_pos = swarm_particles[neighbors[0]].pbest_position
        best_val = swarm_particles[neighbors[0]].pbest_value

        for i in neighbors[1:]:
            p = swarm_particles[i]
            if p.pbest_value < best_val:
                best_val = p.pbest_value
                best_pos = p.pbest_position

        return best_pos