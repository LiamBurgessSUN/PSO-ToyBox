# === Central Hyperparameter Definition ===
# --- Environment Config ---
ENV_DIM = 30
ENV_PARTICLES = 30
ENV_MAX_STEPS = 5000  # Max PSO steps per env run
USE_VELOCITY_CLAMPING = False

# --- Agent/Env Interaction Config ---
AGENT_STEP_SIZE = 125  # Used for fixed Nt mode
ADAPTIVE_NT_MODE = False  # Set to True to enable adaptive Nt
NT_RANGE = (1, 125)  # Range if ADAPTIVE_NT_MODE is True

# --- Training Config ---
EPISODES_PER_FUNCTION = 15  # Renamed from NUM_EPISODES for clarity with train.py arg
BATCH_SIZE = 256
START_STEPS = 100  # Total agent steps before learning starts
UPDATES_PER_STEP = 1  # Agent updates per agent step
SAVE_FREQ_MULTIPLIER = 4  # Renamed from SAVE_FREQ for clarity with train.py arg

# --- Testing Config ---
NUM_EVAL_RUNS = 30  # Number of deterministic runs per test function

# --- RL Model Save/Load Config ---
# Control whether to save, load, or use new RL models
SAVE_RL_MODEL = False          # Whether to save the trained RL model
LOAD_RL_MODEL = False         # Whether to load an existing RL model for training
USE_NEW_MODEL = True          # Whether to use a fresh model (ignored if LOAD_RL_MODEL is True)
MODEL_SAVE_FREQUENCY = 5      # Save model every N functions (if SAVE_RL_MODEL is True)
AUTO_SAVE_FINAL = True        # Always save final model after training (if SAVE_RL_MODEL is True)

# Model file naming and organization
MODEL_NAME_PREFIX = "sapso_agent"  # Prefix for saved model files
MODEL_VERSION_SUFFIX = ""          # Optional version suffix (e.g., "_v1", "_experiment1")
INCLUDE_TIMESTAMP = True           # Include timestamp in model filename
MODEL_METADATA_SAVE = True         # Save training metadata with model

# --- Agent Hyperparameters (can also be passed if needed) ---
# hidden_dim=256; gamma=1.0; tau=0.005; alpha=0.2; actor_lr=3e-4; critic_lr=3e-4
# For simplicity, keeping these inside train/test for now, but could be passed

# --- Checkpoint/Output Config ---
# Construct paths relative to the project root found earlier
# (Checkpoint filenames are constructed inside train/test based on config)
CHECKPOINT_BASE_DIR = "Benchmark/models"  # Relative path for models/checkpoints
