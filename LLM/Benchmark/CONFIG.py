# === Central Hyperparameter Definition ===
# --- Environment Config ---
ENV_DIM = 30
ENV_PARTICLES = 30
ENV_MAX_STEPS = 5000  # Max PSO steps per env run
USE_VELOCITY_CLAMPING = True

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

# --- Agent Hyperparameters (can also be passed if needed) ---
# hidden_dim=256; gamma=1.0; tau=0.005; alpha=0.2; actor_lr=3e-4; critic_lr=3e-4
# For simplicity, keeping these inside train/test for now, but could be passed

# --- Checkpoint/Output Config ---
# Construct paths relative to the project root found earlier
# (Checkpoint filenames are constructed inside train/test based on config)
CHECKPOINT_BASE_DIR = "models"  # Relative path for models/checkpoints
