"""
Interactive visualization using Plotly.
Provides interactive charts, dashboards, and animations.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any

# Try to import Plotly
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Warning: Plotly not available. Interactive visualizations disabled.")

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG


class PlotlyVisualizer:
    """
    Interactive visualization using Plotly.
    Provides rich, interactive visualizations for the simulation.
    """

    def __init__(self, simulation=None, config=None):
        """
        Initialize the Plotly visualizer.

        Args:
            simulation: Simulation object (optional)
            config: Configuration dictionary (optional)
        """
        if not HAS_PLOTLY:
            raise ImportError("Plotly is required for interactive visualization.")

        self.simulation = simulation
        self.config = config or CONFIG
        self.setup_colors()

    def setup_colors(self):
        """Set up color schemes for visualization."""
        # Color schemes
        self.colorscale_communities = px.colors.qualitative.Plotly
        self.colorscale_energy = px.colors.sequential.Viridis
        self.colorscale_resources = px.colors.sequential.Greens
        self.colorscale_alliances = px.colors.sequential.Blues

    def create_world_visualization(self, state):
        """
        Create an interactive visualization of the simulation world.

        Args:
            state: Current simulation state

        Returns:
            Plotly figure
        """
        # Create base figure
        fig = go.Figure()

        # Add resources
        self._add_resources(fig, state["resources"])

        # Add agents by community
        self._add_agents(fig, state["agents"])

        # Add alliances
        self._add_alliances(fig, state["agents"], state["communities"], state["alliances"])

        # Configure layout
        world_size = self.config["world_size"]
        fig.update_layout(
            title=f"Simulation Step {state['step']} - Population: {len(state['agents'])}",
            xaxis=dict(
                title="X Position",
                range=[0, world_size[0]],
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title="Y Position",
                range=[0, world_size[1]],
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                scaleanchor="x",  # Make sure x and y scales are equal
                scaleratio=1
            ),
            hovermode="closest",
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
            margin=dict(l=0, r=0, t=40, b=0),
        )

        # Add environmental annotations
        self._add_environmental_indicators(fig, state["environment"])

        return fig

    def _add_resources(self, fig, resources):
        """
        Add resources to the visualization.

        Args:
            fig: Plotly figure
            resources: List of resource objects
        """
        if not resources:
            return

        positions = np.array([r["position"] for r in resources])
        values = np.array([r["value"] for r in resources])
        ids = [r["id"] for r in resources]

        # Create hover text
        hover_texts = [
            f"Resource {r_id}<br>Value: {value:.1f}"
            for r_id, value in zip(ids, values)
        ]

        # Scale marker size by value
        min_val = min(values)
        max_val = max(values)
        size_scale = lambda v: 5 + 15 * ((v - min_val) / (max_val - min_val + 1e-6))
        sizes = [size_scale(v) for v in values]

        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=positions[:, 0],
            y=positions[:, 1],
            mode='markers',
            marker=dict(
                symbol='square',
                size=sizes,
                color=values,
                colorscale=self.colorscale_resources,
                opacity=0.7,
                colorbar=dict(
                    title="Resource Value",
                    thickness=15,
                    len=0.3,
                    y=0.5,
                    yanchor="middle",
                    x=1.02,
                    xanchor="left"
                )
            ),
            text=hover_texts,
            hoverinfo='text',
            name='Resources'
        ))

    def _add_agents(self, fig, agents):
        """
        Add agents to the visualization grouped by community.

        Args:
            fig: Plotly figure
            agents: List of agent objects
        """
        if not agents:
            return

        # Group by community
        communities = {}
        for agent in agents:
            community_id = agent["community"]
            if community_id not in communities:
                communities[community_id] = []
            communities[community_id].append(agent)

        # Add each community as a separate trace
        for i, (community_id, members) in enumerate(communities.items()):
            positions = np.array([a["position"] for a in members])
            energies = np.array([a["energy"] for a in members])
            agent_ids = [a["id"] for a in members]
            knowledge_counts = [len(a.get("knowledge", [])) for a in members]

            # Create hover text
            hover_texts = [
                f"Agent {a_id}<br>Energy: {e:.1f}<br>Knowledge: {k}<br>Age: {a.get('age', 0)}"
                for a_id, e, k, a in zip(agent_ids, energies, knowledge_counts, members)
            ]

            # Choose color
            if community_id == -1:
                color = 'gray'
                name = 'No Community'
            else:
                color_idx = i % len(self.colorscale_communities)
                color = self.colorscale_communities[color_idx]
                name = f'Community {community_id}'

            # Scale size by energy
            sizes = 8 + 12 * (energies / self.config["max_energy"])

            # Add scatter plot
            fig.add_trace(go.Scatter(
                x=positions[:, 0],
                y=positions[:, 1],
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=color,
                    line=dict(width=1, color='black')
                ),
                text=hover_texts,
                hoverinfo='text',
                name=name
            ))

    def _add_alliances(self, fig, agents, communities, alliances):
        """
        Add alliance lines between communities.

        Args:
            fig: Plotly figure
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

        # Add alliance lines
        for key, alliance in alliances.items():
            c1, c2 = key
            if c1 in community_centers and c2 in community_centers:
                strength = alliance["strength"]
                center1 = community_centers[c1]
                center2 = community_centers[c2]

                # Only draw significant alliances
                if strength > 0.2:
                    fig.add_trace(go.Scatter(
                        x=[center1[0], center2[0]],
                        y=[center1[1], center2[1]],
                        mode='lines',
                        line=dict(
                            color='blue',
                            width=3 * strength,
                            dash='solid'
                        ),
                        opacity=strength,
                        hoverinfo='text',
                        hovertext=f"Alliance: {c1}-{c2}<br>Strength: {strength:.2f}",
                        showlegend=False
                    ))

    def _add_environmental_indicators(self, fig, environment):
        """
        Add environmental indicators to the visualization.

        Args:
            fig: Plotly figure
            environment: Environment state dictionary
        """
        conditions = environment.get("conditions", {})

        # Create annotation text
        annotation_text = []

        if "temperature" in conditions:
            temp = conditions["temperature"]
            annotation_text.append(f"Temperature: {temp:.2f}")

        if "day_night" in conditions:
            day = conditions["day_night"]
            cycle = "Day" if day > 0.5 else "Night"
            annotation_text.append(f"{cycle}: {day:.2f}")

        if "season" in conditions:
            seasons = ["Spring", "Summer", "Fall", "Winter"]
            season = seasons[int(conditions["season"])]
            annotation_text.append(f"Season: {season}")

        if "disaster_risk" in conditions:
            risk = conditions["disaster_risk"]
            if risk > 0.1:
                annotation_text.append(f"Disaster Risk: {risk:.2f}")

        # Add annotation
        if annotation_text:
            fig.add_annotation(
                x=0.01,
                y=0.99,
                xref="paper",
                yref="paper",
                text="<br>".join(annotation_text),
                showarrow=False,
                font=dict(
                    family="Arial",
                    size=12,
                    color="black"
                ),
                align="left",
                bgcolor="white",
                opacity=0.7,
                bordercolor="black",
                borderwidth=1,
                borderpad=4
            )

    def create_stats_dashboard(self, stats):
        """
        Create an interactive dashboard of simulation statistics.

        Args:
            stats: Statistics dictionary

        Returns:
            Plotly figure
        """
        if not stats or not any(len(v) > 0 for v in stats.values() if isinstance(v, list)):
            return None

        # Create a 3x2 subplot grid
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Population", "Average Energy",
                "Communities", "Alliances",
                "Knowledge Discovered", "Additional Metrics"
            ),
            shared_xaxes=True,
            vertical_spacing=0.1
        )

        # Add time axis for all plots
        steps = range(len(next((v for v in stats.values() if isinstance(v, list) and len(v) > 0), [])))

        # Helper function to add a trace
        def add_stat_trace(key, row, col, name=None, color=None, line_shape='linear'):
            if key in stats and len(stats[key]) > 0:
                values = stats[key]
                x_values = steps[:len(values)]

                # Calculate rolling average for smoother visualization
                if len(values) > 10:
                    window_size = max(5, len(values) // 20)
                    rolling_avg = np.convolve(
                        values,
                        np.ones(window_size) / window_size,
                        mode='valid'
                    )
                    roll_x = steps[window_size - 1:len(values)]

                    # Add the rolling average
                    fig.add_trace(
                        go.Scatter(
                            x=roll_x,
                            y=rolling_avg,
                            mode='lines',
                            line=dict(width=3, color=color or 'red'),
                            name=f"{name or key} (trend)",
                            showlegend=False
                        ),
                        row=row,
                        col=col
                    )

                # Add the raw data
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=values,
                        mode='lines',
                        line=dict(
                            color=color,
                            shape=line_shape
                        ),
                        name=name or key,
                        showlegend=True,
                        opacity=0.7
                    ),
                    row=row,
                    col=col
                )

        # Population
        add_stat_trace("population", 1, 1, "Population", "#1f77b4")

        # Average Energy
        add_stat_trace("avg_energy", 1, 2, "Average Energy", "#2ca02c")

        # Communities
        add_stat_trace("community_count", 2, 1, "Communities", "#9467bd")

        # Alliances
        add_stat_trace("alliance_count", 2, 2, "Alliances", "#1f77b4")

        # Knowledge
        add_stat_trace("knowledge_discovered", 3, 1, "Knowledge", "#ff7f0e")

        # Additional metrics in the last plot
        additional_metrics = []

        if "genetic_diversity" in stats and len(stats["genetic_diversity"]) > 0:
            additional_metrics.append(("genetic_diversity", "Genetic Diversity", "#d62728"))

        if "conflict_count" in stats and len(stats["conflict_count"]) > 0:
            additional_metrics.append(("conflict_count", "Conflicts", "#d62728"))

        if "cooperation_events" in stats and len(stats["cooperation_events"]) > 0:
            additional_metrics.append(("cooperation_events", "Cooperation", "#2ca02c"))

        # Add up to 3 additional metrics
        for i, (key, name, color) in enumerate(additional_metrics[:3]):
            add_stat_trace(key, 3, 2, name, color)

        # Update layout
        fig.update_layout(
            title="Simulation Statistics",
            height=800,
            width=1000,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=50, r=50, t=80, b=50),
        )

        # Update axes
        fig.update_xaxes(title_text="Simulation Step", row=3, col=1)
        fig.update_xaxes(title_text="Simulation Step", row=3, col=2)

        return fig

    def create_agent_analysis(self, agents):
        """
        Create interactive visualizations for agent analysis.

        Args:
            agents: List of agent objects

        Returns:
            Dictionary of Plotly figures
        """
        if not agents:
            return {}

        figures = {}

        # Extract agent data
        agent_data = pd.DataFrame([
            {
                "id": agent["id"],
                "energy": agent["energy"],
                "age": agent.get("age", 0),
                "community": agent["community"],
                "knowledge": len(agent.get("knowledge", [])),
                "x": agent["position"][0],
                "y": agent["position"][1],
                **{k: v for k, v in agent.get("genome", {}).items()}
            }
            for agent in agents
        ])

        # Energy distribution by community
        figures["energy_dist"] = px.box(
            agent_data,
            x="community",
            y="energy",
            color="community",
            title="Energy Distribution by Community",
            labels={"community": "Community ID", "energy": "Energy Level"},
            height=500
        )

        # Knowledge vs Age scatter plot
        figures["knowledge_age"] = px.scatter(
            agent_data,
            x="age",
            y="knowledge",
            color="community",
            size="energy",
            hover_data=["id"],
            title="Knowledge vs Age",
            labels={"age": "Age", "knowledge": "Knowledge Count"},
            height=500
        )

        # Genome traits distribution
        genome_traits = [col for col in agent_data.columns if col in [
            "aggression", "cooperation", "curiosity", "social",
            "loyalty", "risk_tolerance", "speed", "strength",
            "perception", "resilience", "learning_ability"
        ]]

        if genome_traits:
            # Parallel coordinates plot for genome traits
            figures["genome"] = px.parallel_coordinates(
                agent_data,
                dimensions=genome_traits,
                color="community",
                title="Genome Traits Distribution",
                height=600
            )

            # Radar chart for average traits by community
            community_traits = agent_data.groupby("community")[genome_traits].mean().reset_index()

            # Prepare data for radar chart
            radar_data = []
            for _, row in community_traits.iterrows():
                comm_id = row["community"]

                # Create a trace for each community
                radar_data.append(
                    go.Scatterpolar(
                        r=[row[trait] for trait in genome_traits],
                        theta=genome_traits,
                        fill='toself',
                        name=f"Community {comm_id}" if comm_id != -1 else "No Community"
                    )
                )

            # Create radar chart
            figures["trait_radar"] = go.Figure(data=radar_data)
            figures["trait_radar"].update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title="Average Traits by Community",
                height=600
            )

        return figures

    def create_community_network(self, agents, communities, relationships):
        """
        Create a network visualization of communities and relationships.

        Args:
            agents: List of agent objects
            communities: List of community sets
            relationships: Dictionary of relationships

        Returns:
            Plotly figure
        """
        try:
            import networkx as nx
        except ImportError:
            print("NetworkX not available. Cannot create community network visualization.")
            return None

        # Create graph
        G = nx.Graph()

        # Add nodes (agents)
        for agent in agents:
            G.add_node(
                agent["id"],
                community=agent["community"],
                energy=agent["energy"],
                knowledge=len(agent.get("knowledge", [])),
                position=agent["position"]
            )

        # Add edges (relationships)
        for (a1, a2), data in relationships.items():
            if a1 in G and a2 in G:
                # Calculate relationship strength
                if "history" in data:
                    history = data["history"]
                    if history:
                        # Use simple average for relationship strength
                        strength = sum(history) / len(history)

                        # Add edge if relationship is significant
                        if abs(strength) > 0.2:
                            G.add_edge(a1, a2, weight=strength, trust=data.get("trust", 0.5))

        # Use Fruchterman-Reingold layout for nice visualization
        pos = nx.spring_layout(G, seed=42)

        # Group by community
        community_nodes = {}
        for node, attr in G.nodes(data=True):
            comm = attr["community"]
            if comm not in community_nodes:
                community_nodes[comm] = []
            community_nodes[comm].append(node)

        # Create figure
        fig = go.Figure()

        # Add edges first (so they're on bottom)
        edge_x = []
        edge_y = []
        edge_colors = []
        edge_widths = []

        for u, v, attr in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]

            # Add line segment
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

            # Set color by relationship type
            if attr["weight"] > 0:
                color = "rgba(0, 128, 0, 0.5)"  # Green for positive
            else:
                color = "rgba(255, 0, 0, 0.5)"  # Red for negative

            edge_colors.append(color)
            edge_widths.append(abs(attr["weight"]) * 3 + 1)

        # Add all edges as one trace
        if edge_x:
            fig.add_trace(go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=0.5, color="lightgray"),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            ))

        # Add nodes for each community
        for i, (comm, nodes) in enumerate(community_nodes.items()):
            node_x = [pos[n][0] for n in nodes]
            node_y = [pos[n][1] for n in nodes]

            # Get node attributes
            energies = [G.nodes[n]["energy"] for n in nodes]
            knowledge = [G.nodes[n]["knowledge"] for n in nodes]

            # Scale node size by energy
            size = [e / 10 + 10 for e in energies]

            # Hover text
            hover_text = [
                f"Agent {n}<br>Community: {G.nodes[n]['community']}<br>" +
                f"Energy: {G.nodes[n]['energy']:.1f}<br>" +
                f"Knowledge: {G.nodes[n]['knowledge']}"
                for n in nodes
            ]

            # Choose color
            if comm == -1:
                color = "gray"
                comm_name = "No Community"
            else:
                color = self.colorscale_communities[i % len(self.colorscale_communities)]
                comm_name = f"Community {comm}"

            # Add nodes
            fig.add_trace(go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers',
                marker=dict(
                    size=size,
                    color=color,
                    line=dict(width=1, color="black")
                ),
                text=hover_text,
                hoverinfo='text',
                name=comm_name
            ))

        # Update layout
        fig.update_layout(
            title="Community Network",
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=800,
            width=1000
        )

        return fig