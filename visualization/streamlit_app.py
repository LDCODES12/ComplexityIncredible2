"""
Interactive Streamlit dashboard for the social evolution simulator.
Provides real-time visualization, controls, and analysis.
"""

import os
import sys
import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
import threading
import queue

# Try to import Streamlit
try:
    import streamlit as st
    import pandas as pd

    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    print("Warning: Streamlit not available. Cannot run interactive dashboard.")

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG
from simulation.simulation import Simulation
from visualization.plotly_vis import PlotlyVisualizer

# Check if Plotly is available
try:
    import plotly.graph_objects as go
    import plotly.express as px

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


class SimulationThread(threading.Thread):
    """
    Thread for running the simulation in background.
    Allows Streamlit to update the UI while simulation runs.
    """

    def __init__(self, simulation, state_queue, max_states=100):
        """
        Initialize the simulation thread.

        Args:
            simulation: Simulation object
            state_queue: Queue for storing simulation states
            max_states: Maximum number of states to keep in memory
        """
        super().__init__()
        self.simulation = simulation
        self.state_queue = state_queue
        self.max_states = max_states
        self.running = True
        self.paused = False

    def run(self):
        """Run the simulation thread."""
        while self.running:
            if not self.paused:
                # Check if simulation should continue
                if not self.simulation.update():
                    self.running = False
                    break

                # Get current state and add to queue
                state = self.simulation.get_state()

                # If queue is full, remove oldest state
                if self.state_queue.qsize() >= self.max_states:
                    try:
                        self.state_queue.get_nowait()
                    except queue.Empty:
                        pass

                # Add new state
                self.state_queue.put(state)

            # Small sleep to prevent high CPU usage
            time.sleep(0.01)

    def stop(self):
        """Stop the simulation thread."""
        self.running = False

    def pause(self):
        """Pause the simulation thread."""
        self.paused = True

    def resume(self):
        """Resume the simulation thread."""
        self.paused = False


def run_streamlit_app(config=None):
    """
    Run the Streamlit dashboard application.

    Args:
        config: Configuration dictionary (optional)
    """
    if not HAS_STREAMLIT:
        print("Error: Streamlit is required for the interactive dashboard.")
        return

    if not HAS_PLOTLY:
        print("Error: Plotly is required for the interactive dashboard.")
        return

    # Use provided config or default
    sim_config = config or CONFIG

    # App logic here - will be executed by streamlit.py
    st.title("Social Evolution Simulator")
    st.sidebar.header("Simulation Controls")

    # Create a session key for the simulation
    if "simulation" not in st.session_state:
        st.session_state.simulation = None
        st.session_state.simulation_thread = None
        st.session_state.state_queue = queue.Queue()
        st.session_state.all_states = []
        st.session_state.current_state = None
        st.session_state.config = sim_config
        st.session_state.visualizer = None
        st.session_state.running = False
        st.session_state.step = 0

    # Sidebar configuration
    with st.sidebar.expander("Simulation Parameters", expanded=True):
        # World parameters
        world_size = st.slider(
            "World Size",
            100, 2000,
            st.session_state.config["world_size"][0],
            100
        )

        initial_population = st.slider(
            "Initial Population",
            10, 1000,
            st.session_state.config["initial_population"],
            10
        )

        # Agent parameters
        st.subheader("Agent Parameters")
        max_energy = st.slider(
            "Max Energy",
            50, 200,
            st.session_state.config["max_energy"],
            10
        )

        mutation_rate = st.slider(
            "Mutation Rate",
            0.01, 0.2,
            st.session_state.config["mutation_rate"],
            0.01
        )

        # Social parameters
        st.subheader("Social Parameters")
        community_threshold = st.slider(
            "Community Threshold",
            0.1, 0.9,
            st.session_state.config["community_threshold"],
            0.1
        )

        alliance_formation_rate = st.slider(
            "Alliance Formation Rate",
            0.1, 0.9,
            st.session_state.config["alliance_formation_rate"],
            0.1
        )

        knowledge_value = st.slider(
            "Knowledge Value",
            0.5, 3.0,
            st.session_state.config["knowledge_value"],
            0.1
        )

        # Performance parameters
        st.subheader("Performance")
        n_threads = st.slider(
            "Threads",
            1, 8,
            st.session_state.config["n_threads"],
            1
        )

        # Update configuration if changed
        if (world_size != st.session_state.config["world_size"][0] or
                initial_population != st.session_state.config["initial_population"] or
                max_energy != st.session_state.config["max_energy"] or
                mutation_rate != st.session_state.config["mutation_rate"] or
                community_threshold != st.session_state.config["community_threshold"] or
                alliance_formation_rate != st.session_state.config["alliance_formation_rate"] or
                knowledge_value != st.session_state.config["knowledge_value"] or
                n_threads != st.session_state.config["n_threads"]):
            # Create updated config
            st.session_state.config = sim_config.copy()
            st.session_state.config.update({
                "world_size": (world_size, world_size),
                "initial_population": initial_population,
                "max_energy": max_energy,
                "mutation_rate": mutation_rate,
                "community_threshold": community_threshold,
                "alliance_formation_rate": alliance_formation_rate,
                "knowledge_value": knowledge_value,
                "n_threads": n_threads
            })

    # Control buttons
    col1, col2, col3 = st.sidebar.columns(3)

    with col1:
        start_button = st.button("Start")

    with col2:
        pause_button = st.button("Pause")

    with col3:
        reset_button = st.button("Reset")

    # Handle button actions
    if start_button:
        if st.session_state.simulation is None or not st.session_state.running:
            # Create new simulation if needed
            if st.session_state.simulation is None:
                st.session_state.simulation = Simulation(st.session_state.config)
                st.session_state.visualizer = PlotlyVisualizer(st.session_state.simulation, st.session_state.config)

            # Create and start thread
            if st.session_state.simulation_thread is None or not st.session_state.simulation_thread.is_alive():
                st.session_state.state_queue = queue.Queue()
                st.session_state.simulation_thread = SimulationThread(
                    st.session_state.simulation,
                    st.session_state.state_queue
                )
                st.session_state.simulation_thread.start()
            elif st.session_state.simulation_thread.paused:
                st.session_state.simulation_thread.resume()

            st.session_state.running = True
            st.sidebar.success("Simulation started.")

    if pause_button and st.session_state.simulation_thread is not None:
        if st.session_state.running:
            st.session_state.simulation_thread.pause()
            st.session_state.running = False
            st.sidebar.info("Simulation paused.")

    if reset_button:
        # Stop existing thread if running
        if st.session_state.simulation_thread is not None:
            st.session_state.simulation_thread.stop()
            st.session_state.simulation_thread = None

        # Create new simulation with current config
        st.session_state.simulation = Simulation(st.session_state.config)
        st.session_state.visualizer = PlotlyVisualizer(st.session_state.simulation, st.session_state.config)
        st.session_state.state_queue = queue.Queue()
        st.session_state.all_states = []
        st.session_state.current_state = None
        st.session_state.running = False
        st.session_state.step = 0

        st.sidebar.success("Simulation reset.")

    # Main content area tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "World View", "Statistics", "Agent Analysis", "Communities"
    ])

    # Get the latest state from the queue
    latest_state = None

    if st.session_state.simulation_thread is not None and st.session_state.running:
        # Get all states from queue
        while not st.session_state.state_queue.empty():
            try:
                state = st.session_state.state_queue.get_nowait()
                st.session_state.all_states.append(state)
                latest_state = state
            except queue.Empty:
                break

        if latest_state is not None:
            st.session_state.current_state = latest_state
            st.session_state.step = latest_state["step"]

    # Display current simulation state
    if st.session_state.current_state is not None:
        current_state = st.session_state.current_state

        # Tab 1: World View
        with tab1:
            # Create world visualization
            if st.session_state.visualizer is not None:
                fig = st.session_state.visualizer.create_world_visualization(current_state)
                st.plotly_chart(fig, use_container_width=True)

            # Display simulation information
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Step", current_state["step"])

            with col2:
                st.metric("Population", len(current_state["agents"]))

            with col3:
                st.metric("Communities", len([c for c in current_state["communities"] if c]))

            with col4:
                st.metric("Alliances", len(current_state["alliances"]))

        # Tab 2: Statistics
        with tab2:
            if st.session_state.simulation is not None and len(st.session_state.simulation.stats["population"]) > 1:
                # Create statistics visualization
                stats_fig = st.session_state.visualizer.create_stats_dashboard(st.session_state.simulation.stats)
                if stats_fig is not None:
                    st.plotly_chart(stats_fig, use_container_width=True)
                else:
                    st.info("Not enough data to display statistics yet.")
            else:
                st.info("Run the simulation for a while to see statistics.")

        # Tab 3: Agent Analysis
        with tab3:
            if current_state["agents"]:
                # Filter options
                filter_options = ["All Agents", "By Community", "By Knowledge Level", "By Energy Level"]
                filter_by = st.selectbox("Filter Agents", filter_options)

                # Filter agents based on selection
                filtered_agents = current_state["agents"]

                if filter_by == "By Community":
                    community_ids = sorted(list(set([a["community"] for a in current_state["agents"]])))
                    selected_community = st.selectbox(
                        "Select Community",
                        community_ids,
                        format_func=lambda x: f"Community {x}" if x >= 0 else "No Community"
                    )

                    filtered_agents = [a for a in current_state["agents"] if a["community"] == selected_community]

                elif filter_by == "By Knowledge Level":
                    min_knowledge = st.slider("Minimum Knowledge", 0, 30, 0)
                    filtered_agents = [
                        a for a in current_state["agents"]
                        if len(a.get("knowledge", [])) >= min_knowledge
                    ]

                elif filter_by == "By Energy Level":
                    min_energy = st.slider(
                        "Minimum Energy",
                        0.0,
                        st.session_state.config["max_energy"],
                        0.0
                    )
                    filtered_agents = [a for a in current_state["agents"] if a["energy"] >= min_energy]

                # Display agent analysis
                if filtered_agents:
                    # Create agent analysis visualizations
                    agent_figs = st.session_state.visualizer.create_agent_analysis(filtered_agents)

                    # Display each figure
                    for title, fig in agent_figs.items():
                        st.subheader(title.replace("_", " ").title())
                        st.plotly_chart(fig, use_container_width=True)

                    # Display agent data table
                    with st.expander("Agent Data Table", expanded=False):
                        agent_data = []
                        for agent in filtered_agents:
                            agent_dict = {
                                "ID": agent["id"],
                                "Energy": f"{agent['energy']:.1f}",
                                "Age": agent.get("age", 0),
                                "Community": agent["community"],
                                "Knowledge": len(agent.get("knowledge", []))
                            }

                            # Add genome traits if available
                            if "genome" in agent:
                                for key, value in agent["genome"].items():
                                    agent_dict[key.capitalize()] = f"{value:.2f}"

                            agent_data.append(agent_dict)

                        # Convert to DataFrame
                        df = pd.DataFrame(agent_data)
                        st.dataframe(df)
                else:
                    st.info("No agents match the filter criteria.")
            else:
                st.info("No agents available for analysis.")

        # Tab 4: Communities
        with tab4:
            if current_state["agents"] and any(current_state["communities"]):
                # Community analysis
                community_counts = [len(c) for c in current_state["communities"] if c]

                if community_counts:
                    # Overview metrics
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Number of Communities", len(community_counts))

                    with col2:
                        st.metric("Largest Community", max(community_counts))

                    with col3:
                        st.metric("Average Size", f"{sum(community_counts) / len(community_counts):.1f}")

                    # Community network visualization
                    if st.session_state.visualizer is not None:
                        st.subheader("Community Network")

                        network_fig = st.session_state.visualizer.create_community_network(
                            current_state["agents"],
                            current_state["communities"],
                            st.session_state.simulation.social_network.relationships
                        )

                        if network_fig is not None:
                            st.plotly_chart(network_fig, use_container_width=True)
                        else:
                            st.info("Network visualization requires NetworkX.")

                    # Community details
                    st.subheader("Community Details")

                    # Select community to view
                    community_options = [-1] + list(range(len(current_state["communities"])))
                    selected_comm = st.selectbox(
                        "Select Community",
                        community_options,
                        format_func=lambda x: f"Community {x}" if x >= 0 else "No Community"
                    )

                    # Display community members
                    if selected_comm >= -1 and selected_comm < len(current_state["communities"]):
                        community = current_state["communities"][selected_comm] if selected_comm >= 0 else []

                        # Get community members
                        members = [a for a in current_state["agents"] if a["id"] in community]

                        if members:
                            # Display community alliance information
                            alliances = []
                            for key, alliance in current_state["alliances"].items():
                                if selected_comm in key:
                                    other_comm = key[0] if key[1] == selected_comm else key[1]
                                    alliances.append({
                                        "Community": other_comm,
                                        "Strength": f"{alliance['strength']:.2f}"
                                    })

                            if alliances:
                                st.subheader("Alliances")
                                st.dataframe(pd.DataFrame(alliances))

                            # Display member information
                            st.subheader(f"Members ({len(members)})")

                            member_data = []
                            for agent in sorted(members, key=lambda a: a["id"]):
                                member_data.append({
                                    "ID": agent["id"],
                                    "Energy": f"{agent['energy']:.1f}",
                                    "Age": agent.get("age", 0),
                                    "Knowledge": len(agent.get("knowledge", []))
                                })

                            st.dataframe(pd.DataFrame(member_data))
                        else:
                            st.info("No members in this community.")
                else:
                    st.info("No communities have formed yet.")
            else:
                st.info("No communities have formed yet.")
    else:
        # No simulation state available
        st.info("Start the simulation to see results.")

    # Display simulation status in sidebar
    with st.sidebar:
        st.subheader("Simulation Status")
        if st.session_state.running:
            st.success("Running")
        elif st.session_state.step > 0:
            st.warning("Paused")
        else:
            st.error("Not Started")

        if st.session_state.step > 0:
            st.text(f"Current Step: {st.session_state.step}")

            if hasattr(st.session_state.simulation, 'start_time'):
                elapsed = time.time() - st.session_state.simulation.start_time
                if elapsed > 0:
                    speed = st.session_state.step / elapsed
                    st.text(f"Speed: {speed:.1f} steps/sec")

    # Add a footer with additional information
    st.markdown("---")
    st.markdown(
        "Social Evolution Simulator - An agent-based model demonstrating emergent social behaviors."
    )


if __name__ == "__main__":
    # Run the Streamlit app directly
    run_streamlit_app()