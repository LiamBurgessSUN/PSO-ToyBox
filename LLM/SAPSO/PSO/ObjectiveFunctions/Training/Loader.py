from .Functions.Ackley import AckleyFunction
from .Functions.Alpine import AlpineFunction
from .Functions.Bohachevsky import BohachevskyFunction
from .Functions.BonyadiMichalewicz import BonyadiMichalewiczFunction
from .Functions.Brown import BrownFunction
from .Functions.CosineMixture import CosineMixtureFunction
from .Functions.DeflectedCorrugatedSpring import DeflectedCorrugatedSpringFunction
from .Functions.Discuss import DiscussFunction
from .Functions.DropWave import DropWaveFunction
from .Functions.EggCrate import EggCrateFunction
from .Functions.EggHolder import EggHolderFunction
from .Functions.Elliptic import EllipticFunction
from .Functions.Exponential import ExponentialFunction
from .Functions.Giunta2D import GiuntaFunction
from .Functions.HolderTable import HolderTable1Function
from .Functions.Levy import Levy3Function
from .Functions.LevyMontalvo import LevyMontalvo2Function
from .Functions.Mishra import Mishra1Function, Mishra4Function
from .Functions.NeedleEye import NeedleEyeFunction
from .Functions.Norweigan import NorwegianFunction  # Corrected class name if needed
from .Functions.Pathological import PathologicalFunction
from .Functions.Penalty import Penalty1Function, Penalty2Function
from .Functions.Periodic import PeriodicFunction
from .Functions.Pinter import PinterFunction
from .Functions.Price import Price2Function
from .Functions.Qings import QingsFunction
from .Functions.Quadratic import QuadricFunction
from .Functions.Quintic import QuinticFunction
from .Functions.Rana import RanaFunction
from .Functions.Rastrgin import RastriginFunction
from .Functions.Ripple import Ripple25Function
from .Functions.Rosenbrock import RosenbrockFunction
from .Functions.Salomon import SalomonFunction
from .Functions.Schubert import Schubert4Function
from .Functions.Schwefel import Schwefel1Function
from .Functions.Sinusoidal import SinusoidalFunction
from .Functions.Step import StepFunction3
from .Functions.Trid import TridFunction
from .Functions.Trigonometric import TrigonometricFunction
from .Functions.Vincent import VincentFunction
from .Functions.Weierstrass import WeierstrassFunction
from .Functions.XinSheYang import XinSheYang1Function, XinSheYang2Function

# --- End Project Imports ---


# --- Manually Create List of Objective Function Classes ---
# (List remains the same)
objective_function_classes = [
    AckleyFunction, AlpineFunction, BohachevskyFunction, BonyadiMichalewiczFunction,
    BrownFunction, CosineMixtureFunction, DeflectedCorrugatedSpringFunction,
    DiscussFunction, DropWaveFunction, EggCrateFunction, EggHolderFunction,
    EllipticFunction, ExponentialFunction, GiuntaFunction, HolderTable1Function,
    Levy3Function, LevyMontalvo2Function, Mishra1Function, Mishra4Function,
    NeedleEyeFunction, NorwegianFunction, PathologicalFunction, Penalty1Function,
    Penalty2Function, PeriodicFunction, PinterFunction, Price2Function,
    QingsFunction, QuadricFunction, QuinticFunction, RanaFunction, RastriginFunction,
    Ripple25Function, RosenbrockFunction, SalomonFunction, Schubert4Function,
    Schwefel1Function, SinusoidalFunction, StepFunction3, TridFunction,
    TrigonometricFunction, VincentFunction, WeierstrassFunction, XinSheYang1Function,
    XinSheYang2Function
]