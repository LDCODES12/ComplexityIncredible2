#!/usr/bin/env python3
"""
Emergency fix for simulation.py by restoring from backup then applying minimal fix.
"""

import os
import sys
import shutil
from pathlib import Path


def restore_from_backup():
    """Try to restore simulation.py from backup if available."""
    sim_path = Path("simulation/simulation.py")
    backup_path = Path("simulation/simulation.py.bak")

    if backup_path.exists():
        print(f"Restoring {sim_path} from backup {backup_path}")
        shutil.copy2(backup_path, sim_path)
        return True
    else:
        print(f"No backup found at {backup_path}")
        return False


def apply_minimal_fix():
    """Apply a minimal fix to simulation.py."""
    sim_path = Path("simulation/simulation.py")

    if not sim_path.exists():
        print(f"Error: {sim_path} not found!")
        return False

    print(f"Reading {sim_path}")
    with open(sim_path, 'r') as f:
        lines = f.readlines()

    # Find the specific line with multiprocessing check
    for i, line in enumerate(lines):
        if "if self.use_multiprocessing:" in line:
            # Get indentation
            indent = line[:line.find("if")]

            # Replace with fixed code using exact same indentation
            new_lines = [
                f"{indent}# First check if Metal is enabled\n",
                f"{indent}if self.use_metal:\n",
                f"{indent}    # Use sequential processing with Metal to avoid segfaults\n",
                f"{indent}    self._update_agents_sequential()\n",
                f"{indent}elif self.use_multiprocessing:\n",
            ]

            # Replace line with fixed code
            lines[i:i + 1] = new_lines

            print(f"Fixed multiprocessing check at line {i + 1}")

            # Write back the file
            with open(sim_path, 'w') as f:
                f.writelines(lines)

            return True

    print("Could not find multiprocessing check!")
    return False


def main():
    """Apply emergency fix."""
    print("Applying emergency fix to simulation.py")

    # First try to restore from backup
    if restore_from_backup():
        print("Successfully restored from backup")
    else:
        print("No backup available, will try to fix existing file")

    # Apply minimal fix
    if apply_minimal_fix():
        print("\n✅ Fix successfully applied!")
        print("\nYou can now run the simulation with:")
        print("  python main.py --use-metal")
        print("\nNote: When using Metal, parallel processing will be disabled")
        print("      but GPU acceleration will still work.\n")
        return 0
    else:
        print("\n❌ Failed to apply fix!")
        print("Please contact the developer for assistance.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())