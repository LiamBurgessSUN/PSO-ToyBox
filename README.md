# PSO-ToyBox

A research-oriented toolbox that combines Particle Swarm Optimization (PSO) with Reinforcement Learning (RL) evaluation. This project implements a **Self-Adaptive PSO (SAPSO)** system where a Soft Actor-Critic (SAC) agent learns to dynamically control PSO parameters for optimal performance across diverse optimization landscapes.

## 🚀 Tech Stack

### Core Technologies
- **Python 3.8+** - Primary programming language
- **PyTorch** - Deep learning framework for RL agent implementation
- **NumPy** - Vectorized computations for efficient PSO operations
- **Gymnasium** - RL environment interface (OpenAI Gym successor)
- **Matplotlib** - Visualization and plotting capabilities

### Key Libraries
- **torch.nn** - Neural network architectures (Actor-Critic networks)
- **torch.optim** - Optimization algorithms (Adam)
- **numpy.random** - Random number generation for PSO
- **pathlib** - Path handling and file operations
- **collections** - Data structures for tracking and metrics

## 🏗️ Project Architecture

```
PSO-ToyBox/
├── SAPSO_AGENT/                 # Main application directory
│   ├── CONFIG.py               # Central hyperparameter configuration
│   ├── Benchmark/              # Training and evaluation pipeline
│   │   ├── train.py           # SAC agent training script
│   │   ├── test.py            # Model evaluation script
│   │   └── benchmark.py       # Main orchestration script
│   ├── SAPSO/                 # Core SAPSO implementation
│   │   ├── Environment/       # RL environment wrapper
│   │   ├── PSO/              # PSO algorithm implementation
│   │   ├── RL/               # Reinforcement learning components
│   │   └── Graphics/         # Visualization utilities
│   └── Logs/                 # Logging system
└── Function_Visualizers/      # Additional visualization tools
```

## 🔧 Core Components

### 1. PSO Implementation (`SAPSO_AGENT/SAPSO/PSO/`)

#### PSOVectorized Class
- **Purpose**: Vectorized PSO implementation using NumPy arrays
- **Key Features**:
  - Boundary constraint handling (infinite fitness for infeasible particles)
  - Velocity clamping with configurable ratio
  - Convergence tracking with multiple criteria
  - Previous position storage for metric calculations

#### Cognitive Strategies
- **GlobalBestStrategy**: Standard gbest PSO topology
- **LBestStrategy**: Local best topology (available)
- **PositionSharing**: Alternative sharing mechanisms

#### Objective Functions
- **Base Class**: `ObjectiveFunction` with abstract methods
- **Training Functions**: 40+ benchmark functions (Ackley, Rastrigin, Rosenbrock, etc.)
- **Testing Functions**: Separate evaluation set
- **Vectorized Evaluation**: `evaluate_matrix()` for batch processing

#### Metrics System
- **SwarmMetrics**: Comprehensive metric calculations
- **Key Metrics**:
  - Swarm diversity (distance-based)
  - Stability ratio (Poli's condition)
  - Infeasible ratio (boundary violations)
  - Average step size and velocity magnitude

### 2. Reinforcement Learning (`SAPSO_AGENT/SAPSO/RL/`)

#### SAC Agent (`ActorCritic/Agent.py`)
- **Architecture**: Soft Actor-Critic with dual Q-networks
- **Features**:
  - Entropy maximization for exploration
  - Clipped double-Q learning
  - Target network soft updates (Polyak averaging)
  - Deterministic and stochastic action selection

#### Actor Network (`ActorCritic/Actor.py`)
- **Purpose**: Policy network for PSO parameter control
- **Output**: Continuous actions for (ω, c₁, c₂, optional Nt)
- **Architecture**: Multi-layer perceptron with tanh activation

#### Critic Networks (`ActorCritic/Critic.py`)
- **Purpose**: Q-value estimation for state-action pairs
- **Architecture**: Dual Q-networks to reduce overestimation bias
- **Input**: State and action concatenation

#### Replay Buffer (`Replay/ReplayBuffer.py`)
- **Purpose**: Experience replay for stable training
- **Features**: Prioritized sampling, device management, capacity control

### 3. Environment (`SAPSO_AGENT/SAPSO/Environment/`)

#### Environment Class
- **Interface**: Gymnasium-compatible RL environment
- **Observation Space**: 4-dimensional normalized state
  - Squashed normalized average velocity
  - Feasible ratio (0-1)
  - Stability ratio (Poli's condition)
  - Percent completion (0-1)
- **Action Space**: 3-4 dimensional continuous actions
  - Fixed Nt mode: (ω, c₁, c₂)
  - Adaptive Nt mode: (ω, c₁, c₂, Nt)

#### Key Methods
- `reset()`: Initialize new episode
- `step(action)`: Execute agent action and return (state, reward, done, info)
- `_rescale_action()`: Convert [-1,1] actions to PSO parameter bounds

### 4. Training Pipeline (`SAPSO_AGENT/Benchmark/`)

#### Training Process (`train.py`)
- **Multi-Function Training**: Iterates through objective functions
- **Episode Management**: Configurable episodes per function
- **Checkpointing**: Regular model saving with configurable frequency
- **Logging**: Comprehensive training metrics and progress tracking

#### Evaluation Process (`test.py`)
- **Deterministic Evaluation**: 30 runs per test function
- **Performance Metrics**: Final gbest values, convergence analysis
- **Visualization**: Automated plotting of results
- **Statistical Analysis**: Mean, std, and confidence intervals

#### Orchestration (`benchmark.py`)
- **Main Entry Point**: Coordinates training and testing
- **Configuration Management**: Uses centralized CONFIG.py
- **Error Handling**: Robust exception handling and logging

### 5. Configuration System (`SAPSO_AGENT/CONFIG.py`)

#### Environment Parameters
```python
ENV_DIM = 30                    # Problem dimension
ENV_PARTICLES = 30              # Swarm size
ENV_MAX_STEPS = 5000           # Maximum PSO steps per episode
USE_VELOCITY_CLAMPING = False   # Velocity clamping toggle
```

#### Agent Parameters
```python
AGENT_STEP_SIZE = 125          # Fixed Nt (internal PSO steps)
ADAPTIVE_NT_MODE = False       # Adaptive Nt control
NT_RANGE = (1, 125)           # Adaptive Nt bounds
```

#### Training Parameters
```python
EPISODES_PER_FUNCTION = 15     # Training episodes per function
BATCH_SIZE = 256              # SAC batch size
START_STEPS = 100             # Initial exploration steps
UPDATES_PER_STEP = 1          # Agent updates per environment step
```

## 🎯 Key Features

### Research-Aligned Implementation
- **Poli's Stability Condition**: Mathematical stability analysis
- **Paper Metrics**: Aligned with academic research standards
- **Boundary Handling**: Infinite fitness for infeasible particles
- **Convergence Criteria**: Multiple convergence detection methods

### Performance Optimizations
- **Vectorized Operations**: NumPy-based efficient computations
- **Matrix Evaluation**: Batch objective function evaluation
- **GPU Support**: PyTorch CUDA integration
- **Memory Management**: Efficient replay buffer implementation

### Extensibility
- **Modular Design**: Clean separation of concerns
- **Plugin Architecture**: Easy addition of new objective functions
- **Configurable Metrics**: Customizable evaluation criteria
- **Visualization Framework**: Extensible plotting system

## 🚀 Usage

### Quick Start
1. **Install Dependencies**:
   ```bash
   pip install torch numpy gymnasium matplotlib
   ```

2. **Run Training**:
   ```bash
   cd SAPSO_AGENT/Benchmark
   python benchmark.py
   ```

3. **Configuration**:
   - Modify `SAPSO_AGENT/CONFIG.py` for hyperparameter tuning
   - Adjust training/testing parameters as needed

### Custom Objective Functions
1. Create new function class inheriting from `ObjectiveFunction`
2. Implement `evaluate()` and `evaluate_matrix()` methods
3. Add to appropriate loader (`Training/Loader.py` or `Testing/Loader.py`)

### Custom Metrics
1. Extend `SwarmMetrics` class
2. Implement new metric calculation methods
3. Update environment observation space if needed

## 📊 Output and Results

### Training Outputs
- **Checkpoints**: Saved model states and final models
- **Training Logs**: Detailed progress and performance metrics
- **Reward Plots**: Training reward progression per function

### Evaluation Outputs
- **Performance Metrics**: Final gbest values and convergence data
- **Statistical Analysis**: Mean, standard deviation, confidence intervals
- **Visualization Plots**: 
  - Parameter evolution over time
  - Swarm diversity analysis
  - Convergence plots
  - Infeasible particle tracking

## 🔬 Research Applications

This toolbox is designed for:
- **Adaptive PSO Research**: Studying parameter adaptation strategies
- **RL in Optimization**: Exploring reinforcement learning for optimization control
- **Benchmark Evaluation**: Comprehensive testing on diverse optimization landscapes
- **Algorithm Comparison**: Standardized evaluation framework

## 📝 Development Notes

### Code Organization
- **Consistent Logging**: Centralized logging system throughout
- **Error Handling**: Comprehensive exception handling and recovery
- **Documentation**: Inline documentation and type hints
- **Testing**: Modular design for unit testing

### Performance Considerations
- **Vectorization**: NumPy operations for efficiency
- **Memory Management**: Efficient data structures and cleanup
- **GPU Utilization**: PyTorch device management
- **Batch Processing**: Matrix operations where possible

### Future Enhancements
- **Multi-Objective PSO**: Extension to multi-objective optimization
- **Distributed Training**: Multi-GPU and distributed computing support
- **Advanced RL Algorithms**: PPO, TD3, or other RL methods
- **Real-World Applications**: Integration with practical optimization problems

---

**Note**: This project is research-oriented and designed for academic use. The modular architecture makes it suitable for extending and adapting to various optimization research scenarios.