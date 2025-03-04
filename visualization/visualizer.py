"""
Base visualization tools for the simulation.
Handles rendering, animation, and exporting visualization artifacts.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
from typing import List, Dict, Tuple, Optional, Union, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG


class Visualizer:
    """
    Base visualization class for the social evolution simulation.
    Uses matplotlib for rendering and animation.
    """

    def __init__(self, simulation=None, config=None):
        """
        Initialize visualizer with simulation.

        Args:
            simulation: Simulation object (optional)
            config: Configuration dictionary (optional)
        """
        self.simulation = simulation
        self.config = config or CONFIG
        self.fig = None
        self.ax = None
        self.cmap = plt.cm.tab20
        self.agent_artists = {}
        self.resource_artists = {}
        self.alliance_artists = []
        self.annotation_artists = []
        self.setup_colors()

    def setup_colors(self):
        """Set up color schemes for visualization."""
        # Community colors
        self.community_colors = self.cmap(np.linspace(0, 1, 20))

        # Resource colors based on value
        self.resource_cmap = plt.cm.Greens

        # Alliance colors with opacity based on strength
        self.alliance_color = 'blue'

        # Status colors
        self.status_cmap = plt.cm.plasma

    def setup_plot(self, figsize=(10, 8)):
        """
        Set up the visualization plot.

        Args:
            figsize: Figure size in inches
        """
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=figsize)

        plt.tight_layout()

        # Set up world boundaries
        world_size = self.config["world_size"]
        self.ax.set_xlim(0, world_size[0])
        self.ax.set_ylim(0, world_size[1])

        # Add grid for better visibility
        self.ax.grid(alpha=0.3)

        # Set title and labels
        self.ax.set_title("Social Evolution Simulation")
        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")

        plt.close()  # Don't display yet

    def update_plot(self, state):
        """
        Update the visualization with current state.

        Args:
            state: Current simulation state

        Returns:
            Matplotlib figure
        """
        if self.fig is None:
            self.setup_plot()

        # Clear previous frame
        self.ax.clear()

        # Set limits
        world_size = self.config["world_size"]
        self.ax.set_xlim(0, world_size[0])
        self.ax.set_ylim(0, world_size[1])

        # Plot resources
        self._plot_resources(state["resources"])

        # Plot agents and communities
        self._plot_agents(state["agents"], state["communities"])

        # Plot alliances
        self._plot_alliances(state["agents"], state["communities"], state["alliances"])

        # Add environmental indicators
        self._plot_environment(state["environment"])

        # Add title and legend
        step = state["step"]
        population = len(state["agents"])
        community_count = len([c for c in state["communities"] if c])
        alliance_count = len(state["alliances"])

        self.ax.set_title(
            f"Step {step} | Population: {population} | "
            f"Communities: {community_count} | Alliances: {alliance_count}"
        )

        # Add legend with reasonable size
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            # Limit legend items if too many
            if len(handles) > 10:
                handles = handles[:10]
                labels = labels[:10]
                labels[-1] = "..."

            self.ax.legend(
                handles,
                labels,
                loc='upper right',
                fontsize=8,
                markerscale=0.7
            )

        return self.fig

    def _plot_resources(self, resources):
        """
        Plot resources on the map.

        Args:
            resources: List of resource objects
        """
        if not resources:
            return

        # Extract resource positions and values
        positions = np.array([r["position"] for r in resources])
        values = np.array([r["value"] for r in resources])

        # Normalize values for color mapping
        if len(values) > 0:
            norm_values = (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-6)
            colors = self.resource_cmap(norm_values)

            # Plot resources with size based on value
            sizes = 10 + 20 * norm_values
            self.ax.scatter(
                positions[:, 0],
                positions[:, 1],
                c=colors,
                s=sizes,
                marker='s',
                alpha=0.7,
                label='Resources',
                edgecolors='none'
            )

    def _plot_agents(self, agents, communities):
        """
        Plot agents colored by community.

        Args:
            agents: List of agent objects
            communities: List of community sets
        """
        if not agents:
            return

        # Group agents by community
        community_agents = {}
        for agent in agents:
            community_id = agent["community"]
            if community_id not in community_agents:
                community_agents[community_id] = []
            community_agents[community_id].append(agent)

        # Plot each community with distinct color
        for community_id, members in community_agents.items():
            # Extract positions and energies
            positions = np.array([a["position"] for a in members])
            energies = np.array([a["energy"] for a in members])

            # Normalize energies for size
            norm_energies = energies / self.config["max_energy"]
            sizes = 20 + 30 * norm_energies

            # Choose color
            if community_id == -1:
                # No community - gray
                color = 'gray'
                label = 'No Community'
            else:
                # Use community color
                color = self.community_colors[community_id % len(self.community_colors)]
                label = f'Community {community_id}'

            # Plot this community
            self.ax.scatter(
                positions[:, 0],
                positions[:, 1],
                color=color,
                s=sizes,
                alpha=0.8,
                edgecolors='black',
                linewidth=0.5,
                label=label
            )

    def _plot_alliances(self, agents, communities, alliances):
        """
        Plot alliances between communities.

        Args:
            agents: List of agent objects
            communities: List of community sets
            alliances: Dictionary of alliances
        """
        if not communities or not alliances:
            return

        # Calculate community centers
        community_centers = {}

        # Get positions of agents in each community
        agent_positions = {agent["id"]: agent["position"] for agent in agents}

        for i, community in enumerate(communities):
            if community:
                positions = [agent_positions[agent_id] for agent_id in community if agent_id in agent_positions]
                if positions:
                    center = np.mean(positions, axis=0)
                    community_centers[i] = center

        # Draw alliances as lines
        for key, alliance in alliances.items():
            c1, c2 = key
            if c1 in community_centers and c2 in community_centers:
                strength = alliance["strength"]
                center1 = community_centers[c1]
                center2 = community_centers[c2]

                self.ax.plot(
                    [center1[0], center2[0]],
                    [center1[1], center2[1]],
                    color=self.alliance_color,
                    alpha=strength,
                    linewidth=2 * strength,
                    linestyle='-',
                    zorder=1
                )

    def _plot_environment(self, environment):
        """
        Plot environmental indicators.

        Args:
            environment: Environment state dictionary
        """
        conditions = environment.get("conditions", {})

        # Create a small info panel
        panel_text = []

        if "temperature" in conditions:
            temp = conditions["temperature"]
            panel_text.append(f"Temp: {temp:.2f}")

        if "day_night" in conditions:
            day = conditions["day_night"]
            cycle = "Day" if day > 0.5 else "Night"
            panel_text.append(f"{cycle}: {day:.2f}")

        if "season" in conditions:
            seasons = ["Spring", "Summer", "Fall", "Winter"]
            season = seasons[int(conditions["season"])]
            panel_text.append(f"Season: {season}")

        if "disaster_risk" in conditions:
            risk = conditions["disaster_risk"]
            if risk > 0.1:
                panel_text.append(f"Risk: {risk:.2f}")

        # Add text box with environment info
        if panel_text:
            info_text = "\n".join(panel_text)
            self.ax.text(
                0.02, 0.98,
                info_text,
                transform=self.ax.transAxes,
                verticalalignment='top',
                bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 5}
            )

    def save_frame(self, state, filename):
        """
        Save current state as an image frame.

        Args:
            state: Current simulation state
            filename: Output filename
        """
        fig = self.update_plot(state)
        fig.savefig(filename, dpi=100, bbox_inches='tight')

    def animate(self, states, output_file=None, fps=10):
        """
        Create animation from a sequence of states.

        Args:
            states: List of simulation states
            output_file: Output filename (optional)
            fps: Frames per second

        Returns:
            Animation object
        """
        if self.fig is None:
            self.setup_plot()

        def update_frame(i):
            return self.update_plot(states[i])

        anim = animation.FuncAnimation(
            self.fig,
            update_frame,
            frames=len(states),
            interval=1000 / fps,
            blit=False
        )

        if output_file:
            # Determine writer based on file extension
            extension = os.path.splitext(output_file)[1].lower()

            if extension == '.mp4':
                writer = animation.FFMpegWriter(
                    fps=fps,
                    metadata=dict(artist='Social Evolution Simulator'),
                    bitrate=1800
                )
            elif extension == '.gif':
                writer = animation.PillowWriter(fps=fps)
            else:
                writer = animation.FFMpegWriter(fps=fps)

            anim.save(output_file, writer=writer)
            print(f"Animation saved to {output_file}")

        return anim

    def plot_stats(self, stats=None):
        """
        Plot simulation statistics.

        Args:
            stats: Statistics dictionary (if None, use simulation stats)

        Returns:
            Matplotlib figure
        """
        if stats is None and self.simulation:
            stats = self.simulation.stats

        if not stats:
            print("No statistics available to plot")
            return None

        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle("Simulation Statistics", fontsize=16)

        # Helper function for plotting
        def plot_stat(ax, key, title, ylabel, color='blue', rolling=False):
            if key in stats and len(stats[key]) > 0:
                x = range(len(stats[key]))
                y = stats[key]

                ax.plot(x, y, color=color, alpha=0.7)

                # Add rolling average for noisy data
                if rolling and len(y) > 10:
                    window_size = max(5, len(y) // 20)
                    rolling_avg = np.convolve(y, np.ones(window_size) / window_size, mode='valid')
                    roll_x = range(window_size - 1, len(y))
                    ax.plot(roll_x, rolling_avg, color='red', linewidth=2, alpha=0.8)

                ax.set_title(title)
                ax.set_xlabel("Simulation Step")
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.3)
            else:
                ax.set_title(f"{title} - No Data")

        # Plot each statistic
        plot_stat(axes[0, 0], "population", "Population", "Count")
        plot_stat(axes[0, 1], "avg_energy", "Average Energy", "Energy", color='green')
        plot_stat(axes[1, 0], "community_count", "Number of Communities", "Count", color='purple')
        plot_stat(axes[1, 1], "alliance_count", "Number of Alliances", "Count", color='blue')
        plot_stat(axes[2, 0], "knowledge_discovered", "Knowledge Discovered", "Count", color='orange')

        # Plot additional statistics if available
        if "genetic_diversity" in stats and len(stats["genetic_diversity"]) > 0:
            plot_stat(axes[2, 1], "genetic_diversity", "Genetic Diversity", "Diversity", color='red')
        elif "conflict_count" in stats and len(stats["conflict_count"]) > 0:
            plot_stat(axes[2, 1], "conflict_count", "Conflict Events", "Count", color='red', rolling=True)

        plt.tight_layout()
        return fig

    def create_heatmap(self, data, title, colormap=plt.cm.viridis):
        """
        Create a heatmap visualization.

        Args:
            data: 2D numpy array of values
            title: Heatmap title
            colormap: Matplotlib colormap

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(data, cmap=colormap, origin='lower')
        ax.set_title(title)

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)

        plt.tight_layout()
        return fig

    def plot_social_network(self, social_graph, figsize=(12, 10)):
        """
        Plot the social network graph.

        Args:
            social_graph: Social graph dictionary
            figsize: Figure size in inches

        Returns:
            Matplotlib figure
        """
        # Try to import networkx
        try:
            import networkx as nx
        except ImportError:
            print("NetworkX not available. Cannot plot social network.")
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create graph
        G = nx.Graph()

        # Add nodes
        for node in social_graph["nodes"]:
            G.add_node(
                node["id"],
                community=node["community"],
                status=node["status"]
            )

        # Add edges
        for edge in social_graph["edges"]:
            G.add_edge(
                edge["source"],
                edge["target"],
                weight=edge["weight"],
                trust=edge.get("trust", 0.5)
            )

        # Get position layout
        pos = nx.spring_layout(G, seed=42)

        # Get communities for coloring
        communities = {}
        for node in social_graph["nodes"]:
            community = node["community"]
            if community not in communities:
                communities[community] = []
            communities[community].append(node["id"])

        # Draw nodes by community
        for i, (comm_id, members) in enumerate(communities.items()):
            color = self.community_colors[i % len(self.community_colors)]
            if comm_id == -1:
                color = 'gray'

            nx.draw_networkx_nodes(
                G, pos,
                nodelist=members,
                node_color=[color] * len(members),
                node_size=[G.nodes[n]["status"] * 300 + 50 for n in members],
                alpha=0.8,
                label=f"Community {comm_id}" if comm_id != -1 else "No Community"
            )

        # Draw edges
        edge_colors = [
            'red' if G.edges[e]["weight"] < 0 else 'green'
            for e in G.edges
        ]
        edge_widths = [
            abs(G.edges[e]["weight"]) * 2 + 0.5
            for e in G.edges
        ]

        nx.draw_networkx_edges(
            G, pos,
            width=edge_widths,
            edge_color=edge_colors,
            alpha=0.6
        )

        # Draw labels for important nodes
        important_nodes = [
            n for n, attr in G.nodes(data=True)
            if attr["status"] > 0.5 or G.degree(n) > 3
        ]

        if important_nodes:
            nx.draw_networkx_labels(
                G, pos,
                {n: str(n) for n in important_nodes},
                font_size=8
            )

        # Add title and legend
        ax.set_title("Social Network")
        ax.legend(loc='upper right')

        # Remove axis
        ax.axis('off')

        return fig