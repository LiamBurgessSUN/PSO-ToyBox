from SAPSO_AGENT.SAPSO.PSO.Cognitive.PositionSharing import KnowledgeSharingStrategy


class GlobalBestStrategy(KnowledgeSharingStrategy):
    def __init__(self, swarm):
        self.swarm = swarm

    def get_best_position(self, particle, swarm_particles):
        return self.swarm.gbest_position