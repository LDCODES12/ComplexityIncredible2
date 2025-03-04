"""
Visualization package for rendering, animating, and interactive exploration.
"""

from visualization.visualizer import Visualizer

# Check if Plotly is available
try:
    from visualization.plotly_vis import PlotlyVisualizer
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Check if Streamlit is available
try:
    import streamlit
    from visualization.streamlit_app import run_streamlit_app
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

__all__ = [
    'Visualizer',
    'HAS_PLOTLY',
    'HAS_STREAMLIT'
]

if HAS_PLOTLY:
    __all__.append('PlotlyVisualizer')

if HAS_STREAMLIT:
    __all__.append('run_streamlit_app')