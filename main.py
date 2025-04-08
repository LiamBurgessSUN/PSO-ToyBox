from LLM.PSO.Cognitive.GBest import GlobalBestStrategy
from LLM.PSO.Cognitive.LBest import LocalBestStrategy
from LLM.PSO.ObjectiveFunctions.Ackley import AckleyFunction
from LLM.PSO.ObjectiveFunctions.Alpine import AlpineFunction
from LLM.PSO.ObjectiveFunctions.Bohachevsky import BohachevskyFunction
from LLM.PSO.ObjectiveFunctions.BonyadiMichalewicz import BonyadiMichalewiczFunction
from LLM.PSO.ObjectiveFunctions.Brown import BrownFunction
from LLM.PSO.ObjectiveFunctions.CosineMixture import CosineMixtureFunction
from LLM.PSO.ObjectiveFunctions.CrossLegTable import CrossLegTableFunction
from LLM.PSO.ObjectiveFunctions.DeflectedCorrugatedSpring import DeflectedCorrugatedSpringFunction
from LLM.PSO.ObjectiveFunctions.Discuss import DiscussFunction
from LLM.PSO.ObjectiveFunctions.DropWave import DropWaveFunction
from LLM.PSO.ObjectiveFunctions.EggCrate import EggCrateFunction
from LLM.PSO.ObjectiveFunctions.EggHolder import EggHolderFunction
from LLM.PSO.ObjectiveFunctions.Elliptic import EllipticFunction
from LLM.PSO.ObjectiveFunctions.Exponential import ExponentialFunction
from LLM.PSO.ObjectiveFunctions.Giunta2D import GiuntaFunction
from LLM.PSO.ObjectiveFunctions.HolderTable import HolderTable1Function
from LLM.PSO.ObjectiveFunctions.Lanczos import Lanczos3Function
from LLM.PSO.ObjectiveFunctions.Rastrgin import RastriginFunction
from LLM.PSO.ObjectiveFunctions.Step import StepFunction3
from LLM.PSO.ObjectiveFunctions.Trigonometric import TrigonometricFunction
from LLM.PSO.ObjectiveFunctions.Weierstrass import WeierstrassFunction
from LLM.PSO.PSO import PSO
from LLM.Visualizer import SwarmVisualizer


if __name__ == "__main__":
    obj_func = TrigonometricFunction(dim=2, num_particles=20)  # Must be 2D
    obj_func.plot_3d_surface()
    swarm = PSO(obj_func, strategy=LocalBestStrategy())
    swarm.strategy.swarm = swarm  # backref

    visualizer = SwarmVisualizer(swarm)

    params = [(0.9, 2.0, 0.5), (0.7, 1.5, 1.5), (0.4, 0.5, 2.5)]
    epoch = [0]

    def pso_step():
        omega, c1, c2 = params[min(epoch[0] // 33, 2)]
        swarm.optimize_step(omega, c1, c2)
        epoch[0] += 1

    visualizer.animate(pso_step, num_steps=100)