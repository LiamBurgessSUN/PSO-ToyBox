from .Functions.CrossLegTable import CrossLegTableFunction
from .Functions.Lanczos import Lanczos3Function
from .Functions.Michalewicz import MichalewiczFunction
from .Functions.Schaffer import Schaffer4Function
from .Functions.SineEnvelope import SineEnvelopeFunction
from .Functions.StretchedVSineWave import StretchedVSineWaveFunction
from .Functions.Wavy import WavyFunction

# --- Manually Create List of Test Objective Function Classes ---
test_objective_function_classes = [
    CrossLegTableFunction, Lanczos3Function, MichalewiczFunction, Schaffer4Function,
    SineEnvelopeFunction, StretchedVSineWaveFunction, WavyFunction
]
