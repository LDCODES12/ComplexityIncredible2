from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

# Define extensions
extensions = [
    Extension(
        "social.interactions",
        ["social/interactions.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
    Extension(
        "simulation.spatial.quadtree",
        ["simulation/spatial/quadtree.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native"],
    ),
]

setup(
    name="social_evolution_simulator",
    version="2.0.0",
    description="Advanced social evolution simulator optimized for M2 MacBooks",
    author="Your Name",
    packages=find_packages(),
    ext_modules=cythonize(extensions, annotate=True, language_level=3),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.60.0",
        "numba>=0.54.0",
        "jax>=0.2.24",
        "jaxlib>=0.1.75",
        "deap>=1.3.1",
        "pygad>=2.16.3",
        "plotly>=5.3.1",
        "streamlit>=1.0.0",
        "scipy>=1.7.0",
        "Cython>=0.29.24",
    ],
    python_requires=">=3.8",
)