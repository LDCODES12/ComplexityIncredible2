# Enhanced Social Evolution Simulator

An advanced multi-agent simulation of evolving social dynamics, optimized for performance with comprehensive optimizations including GPU acceleration on Apple Silicon Macs.

## Overview

This enhanced simulator implements complex social behaviors in autonomous agents with optimized performance using:

- **Cython-compiled** critical calculations
- **JAX-accelerated** neural networks
- **Metal GPU acceleration** for spatial operations (Apple Silicon)
- **Optimized quadtree** spatial partitioning
- **DEAP and PyGAD** evolutionary algorithms
- **Vectorized batch operations**
- **Multi-threaded parallel processing**

Agents develop complex emergent behaviors including:
- Community formation and alliances
- Knowledge discovery and sharing
- Status hierarchies and competition
- Cooperation and conflict dynamics
- Mating with genetic inheritance

## Installation

### Prerequisites

- Python 3.8+ (3.9+ recommended)
- C/C++ compiler (for Cython components)
- macOS with M1/M2/M3 chip for Metal acceleration (optional)

### Setup

1. Clone the repository
```bash
git clone https://github.com/yourusername/social-evolution-simulator.git
cd social-evolution-simulator
```

2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scriptsctivate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Compile Cython extensions using the build script
```bash
python build_extensions.py
```

## Running the Simulator

### Command-line Interface (CLI)

```bash
# Basic run with default settings
python main.py

# Specify population and world size
python main.py --population 200 --world-size 1000
```

### GUI Mode with Animation

```bash
# Run with graphical animation
python main.py --mode gui

# Save the animation to a video file
python main.py --mode gui --save-video simulation.mp4
```

### Interactive Streamlit Dashboard

```bash
# Run with interactive Streamlit dashboard
python main.py --mode streamlit
```

## Performance Tuning

### Apple Silicon Optimization

For best performance on M1/M2 MacBooks:

```bash
python main.py --use-metal --threads 8 --batch-size 128
```

### CPU Optimization

For systems without Metal support:

```bash
python main.py --no-metal --threads 4 --batch-size 64
```

### Debugging

Disable optimizations for easier debugging:

```bash
python main.py --no-metal --no-cython
```

## Project Structure

The simulator is organized into modular components:

```
social_evolution_simulator/
├── main.py                   # Entry point
├── config.py                 # Configuration settings
├── build_extensions.py       # Cython build script
├── simulation/
│   ├── simulation.py         # Main simulation logic
│   └── spatial/
│       ├── quadtree.pyx      # Optimized spatial partitioning
│       ├── grid.py           # Grid-based spatial partitioning
│       └── metal_compute.py  # Metal GPU acceleration
├── agents/
│   ├── agent.py              # Agent behavior and decision-making
│   ├── brain.py              # JAX neural networks
│   └── evolution.py          # DEAP/PyGAD integration
├── social/
│   ├── network.py            # Social relationships
│   ├── relationship.py       # Relationship tracking
│   ├── community.py          # Community formation
│   └── interactions.pyx      # Cython-optimized calculations
├── environment/
│   ├── world.py              # Integrated world environment
│   ├── conditions.py         # Environment and weather
│   └── resources.py          # Resource management
├── knowledge/
│   └── knowledge_system.py   # Knowledge discovery and sharing
└── visualization/
    ├── visualizer.py         # Basic visualization
    ├── plotly_vis.py         # Interactive Plotly visualizations
    └── streamlit_app.py      # Interactive Streamlit dashboard
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Troubleshooting

### Common Issues

1. **Cython compilation errors**: Make sure you have a C/C++ compiler installed. On macOS, install Xcode command line tools. On Windows, install Visual C++ Build Tools.

2. **Metal acceleration errors**: Metal acceleration only works on macOS with Apple Silicon (M1/M2/M3). Use `--no-metal` on other platforms.

3. **OpenMP support**: On macOS, install libomp via Homebrew for OpenMP support: `brew install libomp`

4. **Memory errors with large simulations**: Reduce the world size, population, or batch size to fit within available memory.

### Getting Help

If you encounter issues:

1. Check the logs for specific error messages
2. Verify that all dependencies are installed
3. Try running with `--no-metal --no-cython` to use pure Python implementations

## License

This project is licensed under the MIT License - see the LICENSE file for details.
