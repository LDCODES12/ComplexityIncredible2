#!/usr/bin/env python3
"""
Main entry point for the enhanced Social Evolution Simulator.
Handles command-line arguments, initialization, and running the simulation.
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import configuration
from config import CONFIG

# Import simulation components
from simulation.simulation import Simulation
from visualization.visualizer import Visualizer
from visualization.streamlit_app import run_streamlit_app


def main():
    """Main function to start the simulation."""
    parser = argparse.ArgumentParser(description="Enhanced Social Evolution Simulator")

    # Simulation mode
    parser.add_argument(
        "--mode",
        choices=["cli", "gui", "streamlit"],
        default="cli",
        help="Mode to run the simulator"
    )

    # Basic simulation parameters
    parser.add_argument(
        "--steps",
        type=int,
        default=CONFIG["simulation_steps"],
        help="Number of steps to simulate"
    )

    parser.add_argument(
        "--population",
        type=int,
        default=CONFIG["initial_population"],
        help="Initial population size"
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=CONFIG["world_size"][0],
        help="World size (square)"
    )

    # Output options
    parser.add_argument(
        "--save-stats",
        type=str,
        default=None,
        help="Save statistics to file"
    )

    parser.add_argument(
        "--save-video",
        type=str,
        default=None,
        help="Save simulation video to file"
    )

    parser.add_argument(
        "--save-snapshots",
        action="store_true",
        help="Save snapshots of the simulation state"
    )

    parser.add_argument(
        "--snapshot-dir",
        type=str,
        default="snapshots",
        help="Directory to save snapshots"
    )

    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=100,
        help="Steps between snapshots"
    )

    # Performance options
    parser.add_argument(
        "--threads",
        type=int,
        default=CONFIG["n_threads"],
        help="Number of threads for parallel processing"
    )

    parser.add_argument(
        "--use-metal",
        action="store_true",
        help="Use Metal GPU acceleration (Mac only)"
    )

    parser.add_argument(
        "--no-metal",
        action="store_true",
        help="Disable Metal GPU acceleration"
    )

    parser.add_argument(
        "--use-cython",
        action="store_true",
        help="Use Cython-optimized components"
    )

    parser.add_argument(
        "--no-cython",
        action="store_true",
        help="Disable Cython-optimized components"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=CONFIG["batch_size"],
        help="Batch size for agent processing"
    )

    # Parse arguments
    args = parser.parse_args()

    # Update configuration based on arguments
    config = CONFIG.copy()
    config.update({
        "simulation_steps": args.steps,
        "initial_population": args.population,
        "world_size": (args.world_size, args.world_size),
        "n_threads": args.threads,
        "batch_size": args.batch_size,
    })

    # Handle conflicting Metal options
    if args.use_metal and args.no_metal:
        print("Error: Cannot use both --use-metal and --no-metal")
        return 1

    if args.use_metal:
        config["use_metal"] = True
    elif args.no_metal:
        config["use_metal"] = False

    # Handle conflicting Cython options
    if args.use_cython and args.no_cython:
        print("Error: Cannot use both --use-cython and --no-cython")
        return 1

    if args.use_cython:
        config["use_cython"] = True
    elif args.no_cython:
        config["use_cython"] = False

    # Print configuration
    print("Starting simulation with configuration:")
    print(f"  World size: {config['world_size']}")
    print(f"  Initial population: {config['initial_population']}")
    print(f"  Steps: {config['simulation_steps']}")
    print(f"  Threads: {config['n_threads']}")
    print(f"  Metal acceleration: {'Enabled' if config['use_metal'] else 'Disabled'}")
    print(f"  Cython optimization: {'Enabled' if config['use_cython'] else 'Disabled'}")

    # Run in appropriate mode
    if args.mode == "streamlit":
        print("Starting Streamlit interface...")
        run_streamlit_app(config)
        return 0

    # Create simulation
    sim = Simulation(config)
    visualizer = Visualizer(sim)

    # Prepare for snapshots if requested
    if args.save_snapshots:
        # Create snapshot directory
        snapshot_dir = Path(args.snapshot_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)

        print(f"Saving snapshots to {snapshot_dir} every {args.snapshot_interval} steps")

    if args.mode == "gui":
        # Collect states for animation
        states = []

        def state_callback(state):
            states.append(state.copy())

            # Save snapshot if requested
            if args.save_snapshots and sim.step % args.snapshot_interval == 0:
                snapshot_path = snapshot_dir / f"step_{sim.step:06d}.png"
                visualizer.save_frame(state, snapshot_path)

        # Run simulation with callback
        sim.run(callback=state_callback)

        # Create and show animation
        if args.save_video:
            print(f"Saving video to {args.save_video}")
            visualizer.animate(states, args.save_video)
        else:
            anim = visualizer.animate(states)
            plt.show()

    else:  # CLI mode
        # Set up callback for snapshots
        callback = None

        if args.save_snapshots:
            def snapshot_callback(state):
                if sim.step % args.snapshot_interval == 0:
                    snapshot_path = snapshot_dir / f"step_{sim.step:06d}.png"
                    visualizer.save_frame(state, snapshot_path)

            callback = snapshot_callback

        # Run simulation
        stats = sim.run(callback=callback)

        # Print summary
        print(f"Simulation completed after {sim.step} steps")
        print(f"Final population: {len(sim.agents)}")
        print(f"Number of communities: {len(sim.social_network.communities)}")
        print(f"Number of alliances: {len(sim.social_network.alliances)}")

        # Print performance statistics
        if len(stats["performance"]["update_time"]) > 0:
            avg_update_time = np.mean(stats["performance"]["update_time"])
            print(f"Average update time: {avg_update_time:.4f}s")
            print(f"Average steps per second: {1.0 / avg_update_time:.2f}")

        # Plot and save stats
        fig = visualizer.plot_stats()
        if args.save_stats:
            fig.savefig(args.save_stats)

            # Also save raw statistics as JSON
            stats_path = Path(args.save_stats).with_suffix('.json')

            # Convert numpy values to Python native types for JSON serialization
            def convert_to_json_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.generic):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_json_serializable(v) for v in obj]
                else:
                    return obj

            # Convert statistics
            json_stats = convert_to_json_serializable(stats)

            # Save to file
            with open(stats_path, 'w') as f:
                json.dump(json_stats, f, indent=2)

            print(f"Statistics saved to {args.save_stats} and {stats_path}")
        else:
            plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())