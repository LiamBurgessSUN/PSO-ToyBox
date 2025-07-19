#!/usr/bin/env python3
"""
Script to clear the Figures directory before running benchmarks.
"""

import shutil
from pathlib import Path
from SAPSO_AGENT.CONFIG import CHECKPOINT_BASE_DIR

def clear_figures_directory():
    """Clear the Figures directory."""
    figures_dir = Path(CHECKPOINT_BASE_DIR)
    
    if figures_dir.exists():
        try:
            # Remove all contents of the directory
            shutil.rmtree(figures_dir)
            print(f"✅ Cleared existing Figures directory: {figures_dir}")
        except Exception as e:
            print(f"❌ Failed to clear Figures directory {figures_dir}: {e}")
            return False
    
    # Create the directory if it doesn't exist
    try:
        figures_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created/ensured Figures directory: {figures_dir}")
        return True
    except Exception as e:
        print(f"❌ Failed to create Figures directory {figures_dir}: {e}")
        return False

if __name__ == "__main__":
    print("🧹 Clearing Figures directory...")
    if clear_figures_directory():
        print("✅ Figures directory cleared successfully!")
    else:
        print("❌ Failed to clear Figures directory!")
        exit(1) 