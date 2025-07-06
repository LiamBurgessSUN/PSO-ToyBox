"""
Parameter Update Strategies for Baseline PSO

This module provides an abstract base class and concrete implementations
for different methods of updating PSO control parameters (ω, c₁, c₂).
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, Any, Optional
from pathlib import Path

from SAPSO_AGENT.Logs.logger import log_debug


class ParameterStrategy(ABC):
    """
    Abstract base class for PSO parameter update strategies.
    
    This class defines the interface for different methods of updating
    PSO control parameters (ω, c₁, c₂) during optimization.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the parameter strategy.
        
        Args:
            **kwargs: Strategy-specific parameters
        """
        self.kwargs = kwargs
        self.step_count = 0
        self.module_name = Path(__file__).stem
    
    @abstractmethod
    def get_parameters(self, step: int, max_steps: int, **kwargs) -> Tuple[float, float, float]:
        """
        Get the PSO parameters for the current step.
        
        Args:
            step: Current optimization step
            max_steps: Maximum number of steps
            **kwargs: Additional context information (e.g., metrics)
            
        Returns:
            Tuple of (omega, c1, c2) parameters
        """
        pass
    
    def reset(self):
        """Reset the strategy state."""
        self.step_count = 0
    
    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"{self.__class__.__name__}({self.kwargs})"


class FixedParameterStrategy(ParameterStrategy):
    """
    Fixed parameter strategy - uses constant values throughout optimization.
    
    This is the traditional baseline approach where parameters remain unchanged.
    """
    
    def __init__(self, omega: float = 0.7, c1: float = 1.5, c2: float = 1.5, **kwargs):
        """
        Initialize fixed parameter strategy.
        
        Args:
            omega: Fixed inertia weight
            c1: Fixed cognitive coefficient
            c2: Fixed social coefficient
            **kwargs: Additional parameters (ignored)
        """
        super().__init__(**kwargs)
        self.omega = omega
        self.c1 = c1
        self.c2 = c2
    
    def get_parameters(self, step: int, max_steps: int, **kwargs) -> Tuple[float, float, float]:
        """Return fixed parameters regardless of step."""
        return self.omega, self.c1, self.c2


class LinearDecayStrategy(ParameterStrategy):
    """
    Linear decay strategy for inertia weight with fixed cognitive/social coefficients.
    
    Common approach where ω decreases linearly from ω_max to ω_min.
    """
    
    def __init__(self, omega_max: float = 0.9, omega_min: float = 0.4, 
                 c1: float = 1.5, c2: float = 1.5, **kwargs):
        """
        Initialize linear decay strategy.
        
        Args:
            omega_max: Initial inertia weight
            omega_min: Final inertia weight
            c1: Fixed cognitive coefficient
            c2: Fixed social coefficient
            **kwargs: Additional parameters (ignored)
        """
        super().__init__(**kwargs)
        self.omega_max = omega_max
        self.omega_min = omega_min
        self.c1 = c1
        self.c2 = c2
    
    def get_parameters(self, step: int, max_steps: int, **kwargs) -> Tuple[float, float, float]:
        """Return parameters with linearly decaying omega."""
        if max_steps <= 1:
            omega = self.omega_max
        else:
            progress = step / (max_steps - 1)
            omega = self.omega_max - progress * (self.omega_max - self.omega_min)
        
        return omega, self.c1, self.c2


class ExponentialDecayStrategy(ParameterStrategy):
    """
    Exponential decay strategy for inertia weight.
    
    Uses exponential decay function for smoother parameter transitions.
    """
    
    def __init__(self, omega_max: float = 0.9, omega_min: float = 0.4,
                 c1: float = 1.5, c2: float = 1.5, decay_rate: float = 3.0, **kwargs):
        """
        Initialize exponential decay strategy.
        
        Args:
            omega_max: Initial inertia weight
            omega_min: Final inertia weight
            c1: Fixed cognitive coefficient
            c2: Fixed social coefficient
            decay_rate: Rate of exponential decay
            **kwargs: Additional parameters (ignored)
        """
        super().__init__(**kwargs)
        self.omega_max = omega_max
        self.omega_min = omega_min
        self.c1 = c1
        self.c2 = c2
        self.decay_rate = decay_rate
    
    def get_parameters(self, step: int, max_steps: int, **kwargs) -> Tuple[float, float, float]:
        """Return parameters with exponentially decaying omega."""
        if max_steps <= 1:
            omega = self.omega_max
        else:
            progress = step / (max_steps - 1)
            decay_factor = np.exp(-self.decay_rate * progress)
            omega = self.omega_min + (self.omega_max - self.omega_min) * decay_factor
        
        return omega, self.c1, self.c2


class AdaptiveCognitiveSocialStrategy(ParameterStrategy):
    """
    Adaptive strategy that adjusts cognitive and social coefficients.
    
    Gradually shifts from cognitive to social learning as optimization progresses.
    """
    
    def __init__(self, omega: float = 0.7, c1_max: float = 2.5, c1_min: float = 0.5,
                 c2_min: float = 0.5, c2_max: float = 2.5, **kwargs):
        """
        Initialize adaptive cognitive-social strategy.
        
        Args:
            omega: Fixed inertia weight
            c1_max: Initial cognitive coefficient
            c1_min: Final cognitive coefficient
            c2_min: Initial social coefficient
            c2_max: Final social coefficient
            **kwargs: Additional parameters (ignored)
        """
        super().__init__(**kwargs)
        self.omega = omega
        self.c1_max = c1_max
        self.c1_min = c1_min
        self.c2_min = c2_min
        self.c2_max = c2_max
    
    def get_parameters(self, step: int, max_steps: int, **kwargs) -> Tuple[float, float, float]:
        """Return parameters with adaptive cognitive and social coefficients."""
        if max_steps <= 1:
            c1 = self.c1_max
            c2 = self.c2_min
        else:
            progress = step / (max_steps - 1)
            c1 = self.c1_max - progress * (self.c1_max - self.c1_min)
            c2 = self.c2_min + progress * (self.c2_max - self.c2_min)
        
        return self.omega, c1, c2


class ChaosBasedStrategy(ParameterStrategy):
    """
    Chaos-based parameter strategy using logistic map.
    
    Uses chaotic behavior to maintain diversity in parameter values.
    """
    
    def __init__(self, omega_base: float = 0.7, omega_amplitude: float = 0.3,
                 c1_base: float = 1.5, c1_amplitude: float = 0.5,
                 c2_base: float = 1.5, c2_amplitude: float = 0.5,
                 chaos_param: float = 3.9, **kwargs):
        """
        Initialize chaos-based strategy.
        
        Args:
            omega_base: Base inertia weight
            omega_amplitude: Amplitude of omega variation
            c1_base: Base cognitive coefficient
            c1_amplitude: Amplitude of c1 variation
            c2_base: Base social coefficient
            c2_amplitude: Amplitude of c2 variation
            chaos_param: Chaos parameter for logistic map
            **kwargs: Additional parameters (ignored)
        """
        super().__init__(**kwargs)
        self.omega_base = omega_base
        self.omega_amplitude = omega_amplitude
        self.c1_base = c1_base
        self.c1_amplitude = c1_amplitude
        self.c2_base = c2_base
        self.c2_amplitude = c2_amplitude
        self.chaos_param = chaos_param
        self.chaos_state = 0.5  # Initial chaos state
    
    def get_parameters(self, step: int, max_steps: int, **kwargs) -> Tuple[float, float, float]:
        """Return parameters with chaotic variations."""
        # Update chaos state using logistic map
        self.chaos_state = self.chaos_param * self.chaos_state * (1 - self.chaos_state)
        
        # Generate chaotic variations
        omega_var = self.omega_amplitude * (self.chaos_state - 0.5)
        c1_var = self.c1_amplitude * (self.chaos_state - 0.5)
        c2_var = self.c2_amplitude * (self.chaos_state - 0.5)
        
        omega = np.clip(self.omega_base + omega_var, 0.1, 1.0)
        c1 = np.clip(self.c1_base + c1_var, 0.1, 3.0)
        c2 = np.clip(self.c2_base + c2_var, 0.1, 3.0)
        
        return omega, c1, c2
    
    def reset(self):
        """Reset the strategy state including chaos state."""
        super().reset()
        self.chaos_state = 0.5


class FitnessBasedStrategy(ParameterStrategy):
    """
    Fitness-based adaptive strategy.
    
    Adjusts parameters based on fitness improvement rate and swarm diversity.
    """
    
    def __init__(self, omega_base: float = 0.7, c1_base: float = 1.5, c2_base: float = 1.5,
                 improvement_threshold: float = 1e-6, **kwargs):
        """
        Initialize fitness-based strategy.
        
        Args:
            omega_base: Base inertia weight
            c1_base: Base cognitive coefficient
            c2_base: Base social coefficient
            improvement_threshold: Threshold for considering improvement significant
            **kwargs: Additional parameters (ignored)
        """
        super().__init__(**kwargs)
        self.omega_base = omega_base
        self.c1_base = c1_base
        self.c2_base = c2_base
        self.improvement_threshold = improvement_threshold
        self.previous_gbest = float('inf')
        self.stagnation_count = 0
    
    def get_parameters(self, step: int, max_steps: int, **kwargs) -> Tuple[float, float, float]:
        """Return parameters based on fitness improvement."""
        current_gbest = kwargs.get('gbest_value', float('inf'))
        
        # Check for improvement
        if np.isfinite(current_gbest) and np.isfinite(self.previous_gbest):
            improvement = self.previous_gbest - current_gbest
            if improvement > self.improvement_threshold:
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1
        else:
            self.stagnation_count += 1
        
        # Adjust parameters based on stagnation
        if self.stagnation_count > 10:
            # Increase exploration
            omega = np.clip(self.omega_base * 1.2, 0.1, 1.0)
            c1 = np.clip(self.c1_base * 1.1, 0.1, 3.0)
            c2 = np.clip(self.c2_base * 0.9, 0.1, 3.0)
        else:
            # Normal parameters
            omega = self.omega_base
            c1 = self.c1_base
            c2 = self.c2_base
        
        self.previous_gbest = current_gbest
        return omega, c1, c2
    
    def reset(self):
        """Reset the strategy state."""
        super().reset()
        self.previous_gbest = float('inf')
        self.stagnation_count = 0


class TimeVaryingStrategy(ParameterStrategy):
    """
    Time-varying strategy with sinusoidal parameter variations.
    
    Uses periodic functions to create smooth parameter transitions.
    """
    
    def __init__(self, omega_center: float = 0.7, omega_amplitude: float = 0.2,
                 c1_center: float = 1.5, c1_amplitude: float = 0.3,
                 c2_center: float = 1.5, c2_amplitude: float = 0.3,
                 frequency: float = 1.0, **kwargs):
        """
        Initialize time-varying strategy.
        
        Args:
            omega_center: Center value for omega
            omega_amplitude: Amplitude of omega variation
            c1_center: Center value for c1
            c1_amplitude: Amplitude of c1 variation
            c2_center: Center value for c2
            c2_amplitude: Amplitude of c2 variation
            frequency: Frequency of parameter variation
            **kwargs: Additional parameters (ignored)
        """
        super().__init__(**kwargs)
        self.omega_center = omega_center
        self.omega_amplitude = omega_amplitude
        self.c1_center = c1_center
        self.c1_amplitude = c1_amplitude
        self.c2_center = c2_center
        self.c2_amplitude = c2_amplitude
        self.frequency = frequency
    
    def get_parameters(self, step: int, max_steps: int, **kwargs) -> Tuple[float, float, float]:
        """Return parameters with sinusoidal variations."""
        # Normalized time
        t = step / max_steps if max_steps > 0 else 0
        
        # Sinusoidal variations
        omega_var = self.omega_amplitude * np.sin(2 * np.pi * self.frequency * t)
        c1_var = self.c1_amplitude * np.sin(2 * np.pi * self.frequency * t + np.pi/3)
        c2_var = self.c2_amplitude * np.sin(2 * np.pi * self.frequency * t + 2*np.pi/3)
        
        omega = np.clip(self.omega_center + omega_var, 0.1, 1.0)
        c1 = np.clip(self.c1_center + c1_var, 0.1, 3.0)
        c2 = np.clip(self.c2_center + c2_var, 0.1, 3.0)
        
        return omega, c1, c2


class PaperTimeVaryingStrategy(ParameterStrategy):
    """
    Implements the time-varying parameter strategy from the provided equations.
    """
    def get_parameters(self, step: int, max_steps: int, **kwargs) -> Tuple[float, float, float]:
        t = step
        nt = max_steps if max_steps > 0 else 1  # Avoid division by zero

        omega = 0.4 * ((t - nt) / nt) ** 2 + 0.4
        c1 = -3 * (t / nt) + 3.5
        c2 = 3 * (t / nt) + 0.5

        # Optionally clip to reasonable bounds
        # omega = float(np.clip(omega, 0.0, 1.0))
        # c1 = float(np.clip(c1, 0.0, 3.5))
        # c2 = float(np.clip(c2, 0.0, 3.5))

        return omega, c1, c2

# Strategy registry for easy access
PARAMETER_STRATEGIES = {
    'fixed': FixedParameterStrategy,
    'linear_decay': LinearDecayStrategy,
    'exponential_decay': ExponentialDecayStrategy,
    'adaptive_cognitive_social': AdaptiveCognitiveSocialStrategy,
    'chaos_based': ChaosBasedStrategy,
    'fitness_based': FitnessBasedStrategy,
    'time_varying': TimeVaryingStrategy,
    'paper_time_varying': PaperTimeVaryingStrategy
}


def create_strategy(strategy_name: str, **kwargs) -> ParameterStrategy:
    """
    Factory function to create parameter strategies.
    
    Args:
        strategy_name: Name of the strategy to create
        **kwargs: Strategy-specific parameters
        
    Returns:
        ParameterStrategy instance
        
    Raises:
        ValueError: If strategy_name is not recognized
    """
    if strategy_name not in PARAMETER_STRATEGIES:
        available = list(PARAMETER_STRATEGIES.keys())
        raise ValueError(f"Unknown strategy '{strategy_name}'. Available: {available}")
    
    strategy_class = PARAMETER_STRATEGIES[strategy_name]
    return strategy_class(**kwargs)


def list_available_strategies() -> Dict[str, str]:
    """
    Get a list of available parameter strategies with descriptions.
    
    Returns:
        Dictionary mapping strategy names to descriptions
    """
    return {
        'fixed': 'Fixed parameters throughout optimization (traditional baseline)',
        'linear_decay': 'Linear decay of inertia weight with fixed cognitive/social coefficients',
        'exponential_decay': 'Exponential decay of inertia weight',
        'adaptive_cognitive_social': 'Adaptive cognitive and social coefficients',
        'chaos_based': 'Chaos-based parameter variations using logistic map',
        'fitness_based': 'Fitness improvement-based parameter adaptation',
        'time_varying': 'Time-varying parameters with sinusoidal variations',
        'paper_time_varying': 'Time-varying parameters from the paper'
    } 