# SAPSO Plotting Functionality

This module provides comprehensive plotting capabilities for SAPSO training, based on the plotting functions found in the baseline PSO implementation.

## Overview

The SAPSO plotting functionality allows you to generate the same types of plots as the baseline PSO, including:

- **Parameter Evolution**: Control parameters (ω, c₁, c₂) over optimization steps
- **Parameter Comparison**: Parameter evolution across multiple functions
- **Average Parameters**: Average parameters across all functions
- **Stability Condition**: Stability analysis using Poli's condition
- **Infeasible Particles**: Fraction of infeasible particles over time
- **Average Velocity**: Average particle velocity magnitude (log scale)
- **Swarm Diversity**: Swarm diversity over optimization steps (log scale)

## Files

- `sapso_plotting.py`: Main plotting module with `SAPSOPlotter` class
- `example_plotting_usage.py`: Example script demonstrating usage
- `README_PLOTTING.md`: This documentation file

## Usage

### Basic Usage

```python
from SAPSO_AGENT.SAPSO.Graphics.sapso_plotting import SAPSOPlotter
from SAPSO_AGENT.SAPSO.PSO.Metrics.Metrics import SwarmMetrics

# Initialize plotter
plotter = SAPSOPlotter(checkpoint_base_dir="path/to/plots")

# Generate all plots
plotter.plot_all_metrics(metrics_calculator, save_plots=True, show_plots=False)
```

### Individual Plot Types

```python
# Parameter evolution
plotter.plot_parameter_evolution(metrics_calculator, save_plots=True, show_plots=False)

# Parameter comparison across functions
plotter.plot_parameter_comparison(metrics_calculator, save_plots=True, show_plots=False)

# Average parameters
plotter.plot_average_parameters(metrics_calculator, save_plots=True, show_plots=False)

# Stability condition analysis
plotter.plot_stability_condition(metrics_calculator, save_plots=True, show_plots=False)

# Infeasible particles
plotter.plot_infeasible_particles(metrics_calculator, save_plots=True, show_plots=False)

# Average velocity
plotter.plot_average_velocity(metrics_calculator, save_plots=True, show_plots=False)

# Swarm diversity
plotter.plot_swarm_diversity(metrics_calculator, save_plots=True, show_plots=False)
```

## Integration with SAPSO Training

The plotting functionality is automatically integrated into the SAPSO training loop:

1. **Metrics Collection**: During training, the PSO environment automatically tracks metrics and parameters
2. **Data Storage**: Metrics data is stored in the `SwarmMetrics` calculator
3. **Plot Generation**: At the end of training, plots are automatically generated and saved

### Training Integration

The training function (`SAPSO_AGENT/Benchmark/train.py`) has been updated to:

- Collect metrics data during training
- Initialize the `SAPSOPlotter` at the end of training
- Generate all available plots automatically

### Test Integration

The test function (`SAPSO_AGENT/Benchmark/test.py`) has been updated to:

- Support metric plotting during evaluation
- Generate plots for evaluation results

## Plot Types and Features

### Parameter Evolution Plots
- Shows ω, c₁, c₂ over optimization steps
- Per-function individual plots
- Error bands and statistics

### Parameter Comparison Plots
- Compares parameter evolution across functions
- Separate subplots for each parameter
- Color-coded by function

### Average Parameters Plots
- Single plot with average parameters across all functions
- Bar chart with error bars
- Statistical summary

### Stability Condition Plots
- Shows c₁+c₂ vs stability boundary
- Stability fraction over time
- Per-function and averaged plots

### Infeasible Particles Plots
- Fraction of infeasible particles over time
- Per-function and averaged plots
- Statistical summaries

### Average Velocity Plots
- Average particle velocity magnitude
- Logarithmic scale for better visualization
- Per-function and averaged plots

### Swarm Diversity Plots
- Swarm diversity over optimization steps
- Logarithmic scale
- Per-function and averaged plots

## File Organization

Plots are saved in organized directories:

```
checkpoint_base_dir/
├── parameter_evolution/
├── parameter_comparison/
├── average_parameters/
├── stability_condition/
├── infeasible_particles/
├── average_velocity/
└── swarm_diversity/
```

## File Naming Convention

Plots use timestamped filenames:
- Format: `YYYYMMDD_HHMMSS_plot_name.png`
- Example: `20241201_143022_sapso_parameter_evolution_Rastrigin_Ackley_Sphere.png`

## Comparison with Baseline PSO

The SAPSO plotting functions work the same as baseline PSO:

✅ **Same Visual Style**: Identical plot layouts and styling  
✅ **Same Calculations**: Identical statistical calculations  
✅ **Same Features**: Error bands, statistics boxes, legends  
✅ **Same File Structure**: Organized directory structure  
✅ **Same Naming**: Timestamped filenames  

## Requirements

- `matplotlib`: For plotting
- `numpy`: For numerical operations
- `pathlib`: For file path handling
- SAPSO metrics data from training/evaluation

## Example

Run the example script to see the plotting functionality in action:

```bash
python SAPSO_AGENT/example_plotting_usage.py
```

This will generate sample plots and demonstrate all available plotting functions.

## Troubleshooting

### No Metrics Data Available
- Ensure the PSO environment is properly configured with metrics tracking
- Check that the `SwarmMetrics` calculator is initialized
- Verify that metrics data is being collected during training

### Plot Generation Errors
- Check that the checkpoint directory is writable
- Ensure matplotlib is properly installed
- Verify that the metrics data structure is correct

### Missing Plots
- Check the log output for error messages
- Verify that the metrics calculator has tracking data
- Ensure the function names are properly set

## Future Enhancements

Potential improvements:
- Interactive plots with zoom/pan capabilities
- Export plots in additional formats (PDF, SVG)
- Custom plot styling options
- Real-time plotting during training
- Comparison plots between different training runs 