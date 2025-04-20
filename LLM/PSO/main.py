from LLM.PSO.Cognitive.LBest import LocalBestStrategy
from LLM.PSO.ObjectiveFunctions.Training.Trigonometric import TrigonometricFunction
from LLM.PSO.PSO import PSO
from LLM.Visualizer import SwarmVisualizer

if __name__ == "__main__":
    obj_func = TrigonometricFunction(dim=2, num_particles=20)  # Must be 2D
    obj_func.plot_3d_surface()
    swarm = PSO(obj_func, strategy=LocalBestStrategy())
    swarm.kb_sharing_strat.swarm = swarm  # backref

    visualizer = SwarmVisualizer(swarm)

    params = [(0.9, 2.0, 0.5), (0.7, 1.5, 1.5), (0.4, 0.5, 2.5)]
    epoch = [0]


    def pso_step():
        omega, c1, c2 = params[min(epoch[0] // 33, 2)]
        swarm.optimize_step(omega, c1, c2)
        epoch[0] += 1


    visualizer.animate(pso_step, num_steps=100)
