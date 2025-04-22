# logger.py
# Place this file in a suitable location, e.g., PSO-ToyBox/LLM/utils/logger.py
# Adjust imports in other files based on its final location.

import sys
import datetime
import os

# --- Configuration ---
# Set to False to disable color output (e.g., if logging to a file)
ENABLE_COLOR = True
# Check if running in a non-terminal environment (like some IDEs)
IS_TERMINAL = True
# IS_TERMINAL = sys.stdout.isatty()

DEBUG = False

# --- ANSI Escape Codes ---
class Colors:
    RESET = "\033[0m"
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    WHITE = "\033[0;37m"
    BOLD_BLACK = "\033[1;30m"
    BOLD_RED = "\033[1;31m"
    BOLD_GREEN = "\033[1;32m"
    BOLD_YELLOW = "\033[1;33m"
    BOLD_BLUE = "\033[1;34m"
    BOLD_PURPLE = "\033[1;35m"
    BOLD_CYAN = "\033[1;36m"
    BOLD_WHITE = "\033[1;37m"

# --- Color Mapping ---
COLOR_MAP = {
    "default": Colors.RESET,
    "black": Colors.BLACK,
    "red": Colors.RED,
    "green": Colors.GREEN,
    "yellow": Colors.YELLOW,
    "blue": Colors.BLUE,
    "purple": Colors.PURPLE,
    "cyan": Colors.CYAN,
    "white": Colors.WHITE,
    "bold_black": Colors.BOLD_BLACK,
    "bold_red": Colors.BOLD_RED,
    "bold_green": Colors.BOLD_GREEN,
    "bold_yellow": Colors.BOLD_YELLOW,
    "bold_blue": Colors.BOLD_BLUE,
    "bold_purple": Colors.BOLD_PURPLE,
    "bold_cyan": Colors.BOLD_CYAN,
    "bold_white": Colors.BOLD_WHITE,
    # --- Semantic Mappings ---
    "error": Colors.BOLD_RED,
    "warning": Colors.BOLD_YELLOW,
    "info": Colors.CYAN, # Use cyan for info
    "success": Colors.BOLD_GREEN,
    "debug": Colors.PURPLE, # Use purple for debug
    "critical": Colors.BOLD_RED,
    "header": Colors.BOLD_BLUE, # For section headers
    "detail": Colors.WHITE, # For less important details
}

DEFAULT_COLOR_CODE = Colors.RESET

def log(message: str, module_name: str = "INFO", color_name: str = "default"):
    """
    Prints a formatted log message to the console with color.

    Args:
        message (str): The message to print.
        module_name (str): The name of the calling module/package (e.g., __name__).
                           Defaults to "INFO". Can be automatically inferred if needed.
        color_name (str): The name of the color to use (e.g., "red", "info", "warning").
                          Defaults to "default".
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Determine color code
    color_code = COLOR_MAP.get(color_name.lower(), DEFAULT_COLOR_CODE)
    reset_code = Colors.RESET

    # Disable color if not enabled or not in a terminal
    use_color = ENABLE_COLOR and IS_TERMINAL
    if not use_color:
        color_code = ""
        reset_code = ""

    # Format and print the message
    # Pad module name for alignment (adjust padding as needed)
    padded_module = f"[{module_name:<25}]" # Example padding to 25 chars
    formatted_message = f"{timestamp} {padded_module} {color_code}{message}{reset_code}"

    # Use sys.stdout for printing
    print(formatted_message, file=sys.stdout)
    sys.stdout.flush() # Ensure immediate output

# --- Helper Functions for Common Levels ---

def log_error(message: str, module_name: str = "ERROR"):
    """Logs an error message."""
    log(message, module_name, "error")

def log_warning(message: str, module_name: str = "WARNING"):
    """Logs a warning message."""
    log(message, module_name, "warning")

def log_info(message: str, module_name: str = "INFO"):
    """Logs an informational message."""
    log(message, module_name, "info")

def log_success(message: str, module_name: str = "SUCCESS"):
    """Logs a success message."""
    log(message, module_name, "success")

def log_debug(message: str, module_name: str = "DEBUG"):
    """Logs a debug message."""
    # Example: Could add logic to only print debug if a flag is set
    # if os.environ.get("DEBUG_MODE") == "1":
    if DEBUG:
        log(message, module_name, "debug")

def log_header(message: str, module_name: str = "HEADER"):
    """Logs a header/section title message."""
    log(message, module_name, "header")
