# VS Code Launch Configurations for PSO-ToyBox

This directory contains VS Code launch configurations that allow you to run and debug the PSO-ToyBox project without import issues.

## Quick Start

1. **Open the project in VS Code**
2. **Go to the Run and Debug panel** (Ctrl+Shift+D)
3. **Select a configuration from the dropdown** and click the play button

## Available Configurations

### Main Project Scripts
- **Run Benchmark (Training + Testing)**: Complete training and evaluation pipeline
- **Run Training Only**: Train the SAC agent on objective functions
- **Run Testing Only**: Evaluate a trained model
- **Run Function Visualizer**: Generate 3D plots of objective functions

### Debug Scripts
- **Debug Environment**: Test the PSO environment step-by-step
- **Debug SAC Agent**: Test the reinforcement learning agent
- **Debug Current File**: Debug any Python file you have open

### Test Scripts
- **Run PSO Vectorized Test**: Test the core PSO implementation
- **Run Metrics Test**: Test the metrics calculation system

### General Purpose
- **Run Current File**: Run any Python file with proper imports

## Key Features

### Automatic Import Resolution
All configurations set `PYTHONPATH` to the workspace root, so you can import from any module without path issues:

```python
from SAPSO_AGENT.SAPSO.PSO.PSO import PSOVectorized
from SAPSO_AGENT.SAPSO.RL.ActorCritic.Agent import SACAgent
# etc.
```

### Proper Working Directory
All scripts run from the workspace root (`${workspaceFolder}`), ensuring consistent file paths.

### Debug Support
- Set breakpoints in any Python file
- Use "Debug Current File" for step-by-step debugging
- Inspect variables and call stack

## Usage Examples

### Running the Full Pipeline
1. Select "Run Benchmark (Training + Testing)"
2. Click the play button
3. Monitor the training progress in the terminal

### Debugging a Specific Component
1. Open the file you want to debug
2. Set breakpoints by clicking in the gutter
3. Select "Debug Current File"
4. Use F10 (step over) and F11 (step into) to navigate

### Testing Individual Components
1. Select the appropriate test configuration
2. Run to verify the component works correctly
3. Check the output for any errors or warnings

## Troubleshooting

### Import Errors
- Ensure you're using one of the launch configurations
- Check that the file is in the correct location
- Verify the module structure matches the imports

### Path Issues
- All configurations use `${workspaceFolder}` as the working directory
- File paths should be relative to the project root
- Use `os.path.join()` for cross-platform compatibility

### Debug Issues
- Use "Debug Current File" for step-by-step debugging
- Set `justMyCode: false` to debug into library code
- Check the Debug Console for variable inspection

## Customization

You can modify the launch configurations in `.vscode/launch.json`:

- Add new configurations for specific scripts
- Modify environment variables
- Change the Python interpreter
- Add command-line arguments

Example of adding arguments:
```json
{
    "name": "Custom Training",
    "type": "python",
    "request": "launch",
    "program": "${workspaceFolder}/SAPSO_AGENT/Benchmark/train.py",
    "args": ["--episodes", "10", "--batch-size", "128"],
    "console": "integratedTerminal",
    "cwd": "${workspaceFolder}",
    "env": {
        "PYTHONPATH": "${workspaceFolder}"
    }
}
``` 