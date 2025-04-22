# File: PSO-ToyBox/LLM/PSO/Metrics/SwarmMetricsVectorized.py
# Updated to align metric calculations more closely with definitions in
# mathematics-12-03481.pdf (Section 4.1) and accept necessary parameters.
# Added NaN check for input positions.

import numpy as np
import traceback  # For logging exceptions
from pathlib import Path  # To get module name

# --- Import Logger ---
try:
    from ...Logs import logger # Adjust relative path if needed
    from ...Logs.logger import log_info, log_error, log_warning, log_success, log_header, log_debug
except ImportError:
    print("ERROR: Logger module not found at 'LLM.Logs.logger'. Please check path.")
    print("Falling back to standard print statements.")
    def log_info(msg, mod): print(f"INFO [{mod}]: {msg}")
    def log_error(msg, mod): print(f"ERROR [{mod}]: {msg}")
    def log_warning(msg, mod): print(f"WARNING [{mod}]: {msg}")
    def log_success(msg, mod): print(f"SUCCESS [{mod}]: {msg}")
    def log_header(msg, mod): print(f"HEADER [{mod}]: {msg}")
    def log_debug(msg, mod): print(f"DEBUG [{mod}]: {msg}") # Optional debug

# --- Module Name for Logging ---
module_name = Path(__file__).stem # Gets 'SwarmMetricsVectorized'


class SwarmMetricsVectorized:
    """
    Calculates swarm metrics using vectorized NumPy operations, aligned with
    definitions in mathematics-12-03481.pdf where possible.
    """
    def __init__(self):
        """
        Initializes the vectorized metrics calculator.
        """
        pass

    def _check_poli_stability(self, omega: float, c1: float, c2: float) -> bool:
        """
        Checks if the given control parameters satisfy Poli's stability condition (Eq. 4).
        """
        if not (-1.0 <= omega <= 1.0):
            return False
        denominator = 7.0 - 5.0 * omega
        if np.isclose(denominator, 0):
             return False
        stability_boundary = 24.0 * (1.0 - omega**2) / denominator
        is_stable = (c1 + c2) < stability_boundary
        return is_stable


    def compute(self, positions: np.ndarray, previous_positions: np.ndarray,
                velocities: np.ndarray, bounds: tuple,
                omega: float, c1: float, c2: float) -> dict:
        """
        Computes various swarm metrics from the provided state arrays and parameters.
        """
        # --- Input Validation ---
        if positions.shape[0] == 0 or velocities.shape[0] == 0 or previous_positions is None or positions.shape != previous_positions.shape:
             log_warning("Compute called with empty or mismatched arrays.", module_name)
             # Return default dictionary structure with NaN values
             return {
                 'avg_step_size': np.nan,
                 'avg_current_velocity_magnitude': np.nan,
                 'swarm_diversity': np.nan,
                 'infeasible_ratio': np.nan,
                 'stability_ratio': np.nan
             }

        # Check for NaNs in input positions, which would corrupt diversity calculation
        if np.isnan(positions).any():
            log_warning("NaN values detected in input 'positions' array. Diversity will be NaN.", module_name)
            # Proceed with other calculations, but diversity will likely be NaN

        num_particles = positions.shape[0]
        metrics = {}

        # 1. Average Step Size (Paper's "Average particle velocity" - Eq. 29)
        try:
            step_sizes = np.linalg.norm(positions - previous_positions, axis=1)
            metrics['avg_step_size'] = np.mean(step_sizes)
        except Exception as e:
            log_warning(f"Error calculating avg_step_size: {e}", module_name)
            metrics['avg_step_size'] = np.nan


        # 2. Average Current Velocity Magnitude (For comparison/debugging)
        try:
            velocity_magnitudes = np.linalg.norm(velocities, axis=1)
            metrics['avg_current_velocity_magnitude'] = np.mean(velocity_magnitudes)
        except Exception as e:
            log_warning(f"Error calculating avg_current_velocity_magnitude: {e}", module_name)
            metrics['avg_current_velocity_magnitude'] = np.nan


        # 3. Stability Ratio (Poli's Condition - Eq. 4)
        try:
            is_stable = self._check_poli_stability(omega, c1, c2)
            metrics['stability_ratio'] = 1.0 if is_stable else 0.0
        except Exception as e:
            log_warning(f"Error calculating stability_ratio: {e}", module_name)
            metrics['stability_ratio'] = np.nan


        # 4. Swarm Diversity (Matches Paper Definition - Eq. 27)
        metrics['swarm_diversity'] = np.nan # Default to NaN
        if num_particles > 1:
            try:
                # Check again for NaNs specifically before diversity calc
                if not np.isnan(positions).any():
                    centroid = np.mean(positions, axis=0)
                    if not np.isnan(centroid).any(): # Ensure centroid is valid
                        distances = np.linalg.norm(positions - centroid, axis=1)
                        metrics['swarm_diversity'] = np.mean(distances)
                    else:
                        log_warning("Centroid calculation resulted in NaN. Diversity set to NaN.", module_name)
                # else: NaN already logged above
            except Exception as e:
                 log_warning(f"Error calculating swarm_diversity: {e}", module_name)
                 # metrics['swarm_diversity'] remains NaN
        elif num_particles == 1:
             metrics['swarm_diversity'] = 0.0 # Defined as 0 for single particle

        # 5. Infeasible Ratio (Matches Paper Definition)
        try:
            lower_bound, upper_bound = bounds
            is_out_of_bounds = np.any((positions < lower_bound) | (positions > upper_bound), axis=1)
            infeasible_count = np.sum(is_out_of_bounds)
            metrics['infeasible_ratio'] = infeasible_count / num_particles
        except Exception as e:
            log_warning(f"Error calculating infeasible_ratio: {e}", module_name)
            metrics['infeasible_ratio'] = np.nan


        # Final check if any metric is NaN before returning
        if any(np.isnan(v) for v in metrics.values()):
             log_debug(f"Metrics computed with NaNs: {metrics}", module_name)
        else:
             log_debug(f"Computed metrics: {metrics}", module_name)

        return metrics

# # === Example Usage (Remains the same) ===
# if __name__ == "__main__":
#     # ... (example usage code) ...
