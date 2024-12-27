import logging
import networkx as nx
import numpy as np  # type: ignore
import os
from datetime import datetime
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from typing import Dict, List, Optional, Tuple, Any, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeGNNVisualizer:
    def __init__(self):
        """Initialize visualizer with seaborn style settings"""
        self.base_dir = 'visualizations'
        os.makedirs(self.base_dir, exist_ok=True)

        # Create subdirectories for different types of visualizations
        self.dirs = {
            'graphs': os.path.join(self.base_dir, 'graphs'),
            'timeseries': os.path.join(self.base_dir, 'timeseries'),
            'patterns': os.path.join(self.base_dir, 'patterns')
        }

        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)

        # Set seaborn style for better visualizations
        sns.set_theme(style="whitegrid", context="paper")
        plt.style.use('seaborn-v0_8-darkgrid')

    def _create_graph_layout(self, G: nx.Graph) -> Dict:
        """Create a consistent graph layout with proper positioning"""
        try:
            if not G.nodes():
                raise ValueError("Graph has no nodes")

            # Use spring layout with optimized parameters for better visualization
            pos = nx.spring_layout(
                G,
                k=1.5,  # Node spacing
                iterations=50,  # More iterations for better layout
                seed=42  # Fixed seed for consistency
            )
            return pos
        except Exception as e:
            logger.error(f"Error creating graph layout: {str(e)}")
            return {}

    def plot_entity_graph(self, entities_df: pd.DataFrame, 
                         relationships_df: pd.DataFrame,
                         anomalies: Optional[Dict[str, str]] = None,
                         title: Optional[str] = None) -> Optional[plt.Figure]:
        """Plot entity relationship graph focusing only on anomalous nodes and their direct connections"""
        try:
            if not isinstance(entities_df, pd.DataFrame) or not isinstance(relationships_df, pd.DataFrame):
                raise ValueError("Input must be pandas DataFrames")

            if entities_df.empty or relationships_df.empty:
                raise ValueError("Input DataFrames cannot be empty")

            plt.clf()
            fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')

            G = nx.Graph()

            # Only add nodes that are anomalous or connected to anomalous nodes
            if anomalies:
                anomalous_nodes = set(anomalies.keys())
                connected_nodes = set()

                # Find nodes directly connected to anomalous nodes
                for _, row in relationships_df.iterrows():
                    source, target = str(row['source']), str(row['target'])
                    if source in anomalous_nodes:
                        connected_nodes.add(target)
                    if target in anomalous_nodes:
                        connected_nodes.add(source)

                # Add all relevant nodes
                nodes_to_add = anomalous_nodes.union(connected_nodes)
                for _, row in entities_df.iterrows():
                    entity_id = str(row['entity_id'])
                    if entity_id in nodes_to_add:
                        G.add_node(entity_id)

                # Add edges between included nodes
                for _, row in relationships_df.iterrows():
                    source, target = str(row['source']), str(row['target'])
                    if source in G.nodes() and target in G.nodes():
                        G.add_edge(source, target)

                if not G.nodes():
                    logger.info("No anomalies to visualize")
                    return None

                # Get optimized layout
                pos = self._create_graph_layout(G)
                if not pos:
                    raise ValueError("Failed to create graph layout")

                # Draw edges
                nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', alpha=0.6, width=1.5)

                # Draw connected (non-anomalous) nodes
                normal_nodes = [n for n in G.nodes() if n not in anomalous_nodes]
                if normal_nodes:
                    nx.draw_networkx_nodes(G, pos,
                                         nodelist=normal_nodes,
                                         node_color='lightgray',
                                         node_size=300,
                                         alpha=0.5,
                                         ax=ax)
                    nx.draw_networkx_labels(G, pos,
                                          {n: f"E{n}" for n in normal_nodes},
                                          font_size=8)

                # Draw anomalous nodes with their patterns
                nx.draw_networkx_nodes(G, pos,
                                     nodelist=list(anomalous_nodes),
                                     node_color='red',
                                     node_size=500,
                                     alpha=0.8,
                                     ax=ax)
                nx.draw_networkx_labels(G, pos,
                                      {n: f"E{n}\n{anomalies[n]}" for n in anomalous_nodes},
                                      font_size=8,
                                      font_weight='bold')

                # Set title
                plt.title(title or "Anomaly Detection Graph", pad=20, fontsize=14)
                plt.axis('off')
                fig.tight_layout()
                return fig

            logger.info("No anomalies provided for visualization")
            return None

        except Exception as e:
            logger.error(f"Error in plot_entity_graph: {str(e)}")
            return None

    def plot_full_entity_graph(self, entities_df: pd.DataFrame, 
                             relationships_df: pd.DataFrame,
                             title: Optional[str] = None) -> Optional[plt.Figure]:
        """Plot complete entity relationship graph without anomaly highlighting"""
        try:
            if not isinstance(entities_df, pd.DataFrame) or not isinstance(relationships_df, pd.DataFrame):
                raise ValueError("Input must be pandas DataFrames")

            if entities_df.empty or relationships_df.empty:
                raise ValueError("Input DataFrames cannot be empty")

            plt.clf()
            fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')

            G = nx.Graph()

            # Add all nodes
            for _, row in entities_df.iterrows():
                G.add_node(str(row['entity_id']))

            # Add all edges
            for _, row in relationships_df.iterrows():
                source, target = str(row['source']), str(row['target'])
                if source in G.nodes() and target in G.nodes():
                    G.add_edge(source, target)

            if not G.nodes():
                raise ValueError("No valid nodes found in the graph")

            # Get optimized layout
            pos = self._create_graph_layout(G)
            if not pos:
                raise ValueError("Failed to create graph layout")

            # Draw edges
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', alpha=0.6, width=1.5)

            # Draw all nodes
            nx.draw_networkx_nodes(G, pos,
                                 node_color='lightblue',
                                 node_size=300,
                                 alpha=0.8,
                                 ax=ax)
            nx.draw_networkx_labels(G, pos,
                                  {n: f"E{n}" for n in G.nodes()},
                                  font_size=8)

            plt.title(title or "Complete Entity Graph", pad=20, fontsize=14)
            plt.axis('off')
            fig.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Error in plot_full_entity_graph: {str(e)}")
            return None

    def plot_time_series(self, df: pd.DataFrame, entity_id: str) -> Optional[plt.Figure]:
        """Plot time series data using seaborn styling"""
        try:
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")

            if df.empty or 'entity_id' not in df.columns:
                raise ValueError("Invalid DataFrame format")

            entity_data = df[df['entity_id'] == entity_id].copy()
            if entity_data.empty:
                raise ValueError(f"No data found for entity_id: {entity_id}")

            # Set up the figure
            plt.clf()
            fig = plt.figure(figsize=(15, 10), facecolor='white')
            gs = plt.GridSpec(2, 2, figure=fig)

            # Ensure timestamp is datetime
            entity_data['timestamp'] = pd.to_datetime(entity_data['timestamp'])

            # Define metrics and their plots
            metrics = {
                'CPUUtilization': ('CPU Utilization (%)', gs[0, 0]),
                'MemoryUtilization': ('Memory Utilization (%)', gs[0, 1]),
                'NetworkIn': ('Network In (Bytes)', gs[1, 0]),
                'NetworkOut': ('Network Out (Bytes)', gs[1, 1])
            }

            # Create subplots with seaborn styling
            for metric, (title, pos) in metrics.items():
                ax = fig.add_subplot(pos)
                sns.lineplot(data=entity_data, x='timestamp', y=metric, 
                           ax=ax, marker='o', markersize=4)

                ax.set_title(title, fontsize=12, pad=10)
                ax.set_xlabel('Time', fontsize=10)
                ax.set_ylabel(metric.split('(')[0].strip(), fontsize=10)

                # Rotate x-axis labels for better readability
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

                # Add grid for better readability
                ax.grid(True, alpha=0.3)

            plt.suptitle(f"Time Series Metrics for Entity {entity_id}", 
                        y=1.02, fontsize=14, fontweight='bold')
            fig.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Error plotting time series: {str(e)}")
            return None

    def plot_combined_analysis(self, entities_df: pd.DataFrame, 
                         relationships_df: pd.DataFrame,
                         timeseries_df: pd.DataFrame,
                         anomalies: Optional[Dict[str, str]] = None,
                         title: Optional[str] = None) -> Optional[plt.Figure]:
        """Plot combined analysis focusing on anomalies in a single comprehensive view"""
        try:
            if not all([isinstance(df, pd.DataFrame) for df in [entities_df, relationships_df, timeseries_df]]):
                raise ValueError("All inputs must be pandas DataFrames")

            if any(df.empty for df in [entities_df, relationships_df, timeseries_df]):
                raise ValueError("Input DataFrames cannot be empty")

            # Create figure with custom layout
            plt.clf()
            fig = plt.figure(figsize=(20, 12), facecolor='white')
            gs = plt.GridSpec(2, 2, figure=fig, height_ratios=[1.5, 1])

            # Define colors for different patterns
            pattern_colors = {
                'spike': 'red',
                'drop': 'blue',
                'trend': 'purple',
                'seasonal_break': 'orange',
                'gradual': 'green',
                'pattern': 'yellow'
            }

            # 1. Anomaly Graph (Main Focus) - Larger size at top
            ax_graph = fig.add_subplot(gs[0, :])
            G = nx.Graph()

            if anomalies:
                # Add anomalous nodes and their direct neighbors
                anomalous_nodes = set(anomalies.keys())
                connected_nodes = set()

                for _, row in relationships_df.iterrows():
                    source, target = str(row['source']), str(row['target'])
                    if source in anomalous_nodes or target in anomalous_nodes:
                        G.add_node(source)
                        G.add_node(target)
                        G.add_edge(source, target)
                        connected_nodes.add(source)
                        connected_nodes.add(target)

                pos = self._create_graph_layout(G)

                # Draw edges with curved style for better visibility
                nx.draw_networkx_edges(G, pos, ax=ax_graph, 
                                     edge_color='gray', alpha=0.4, 
                                     width=1.5, style='dashed')

                # Draw non-anomalous connected nodes
                normal_nodes = [n for n in G.nodes() if n not in anomalous_nodes]
                if normal_nodes:
                    nx.draw_networkx_nodes(G, pos,
                                       nodelist=normal_nodes,
                                       node_color='lightgray',
                                       node_size=300,
                                       alpha=0.4,
                                       ax=ax_graph)
                    nx.draw_networkx_labels(G, pos,
                                       {n: f"E{n}" for n in normal_nodes},
                                       font_size=8)

                # Draw anomalous nodes with pattern-specific colors
                for pattern in set(anomalies.values()):
                    pattern_nodes = [n for n, p in anomalies.items() if p == pattern]
                    if pattern_nodes:
                        color = pattern_colors.get(pattern, 'gray')
                        nx.draw_networkx_nodes(G, pos,
                                           nodelist=pattern_nodes,
                                           node_color=color,
                                           node_size=500,
                                           alpha=0.8,
                                           ax=ax_graph)
                        nx.draw_networkx_labels(G, pos,
                                           {n: f"E{n}\n{pattern}" for n in pattern_nodes},
                                           font_size=8,
                                           font_weight='bold')

                # Add pattern legend
                pattern_handles = [plt.scatter([], [], c=color, alpha=0.8, s=100, label=pattern.title())
                                for pattern, color in pattern_colors.items()
                                if any(p == pattern for p in anomalies.values())]
                if pattern_handles:  # Only add legend if there are patterns to show
                    ax_graph.legend(handles=pattern_handles, 
                                title='Anomaly Patterns',
                                loc='center left',
                                bbox_to_anchor=(1.02, 0.5))

            ax_graph.set_title("Anomaly Detection Graph", pad=20, fontsize=14)
            ax_graph.axis('off')

            # 2. Time Series Analysis (Bottom Left)
            ax_ts = fig.add_subplot(gs[1, 0])
            ax_memory = None
            ax_network = None

            if anomalies:
                metrics = {
                    'CPUUtilization': {'label': 'CPU (%)', 'color': 'tab:blue', 'linestyle': '-'},
                    'MemoryUtilization': {'label': 'Memory (%)', 'color': 'tab:orange', 'linestyle': '--'},
                    'NetworkIn': {'label': 'Network In (KB/s)', 'color': 'tab:green', 'linestyle': '-.'},
                    'NetworkOut': {'label': 'Network Out (KB/s)', 'color': 'tab:red', 'linestyle': ':'}
                }

                # Create twin axes for different scales
                ax_memory = ax_ts.twinx()
                ax_network = ax_ts.twinx()

                # Offset the right twin axis
                if ax_network is not None:
                    ax_network.spines['right'].set_position(('outward', 60))

                # Plot time series for top anomalous entities
                all_lines = []
                all_labels = []

                for entity_id in list(anomalies.keys())[:3]:
                    entity_data = timeseries_df[timeseries_df['entity_id'] == entity_id].copy()
                    if not entity_data.empty:
                        # Plot each metric
                        for metric, style in metrics.items():
                            if metric in ['CPUUtilization', 'MemoryUtilization']:
                                ax_target = ax_ts if metric == 'CPUUtilization' else ax_memory
                            else:
                                ax_target = ax_network
                                # Convert bytes to KB for network metrics
                                entity_data[metric] = entity_data[metric] / 1024

                            if ax_target is not None:
                                line = ax_target.plot(entity_data['timestamp'],
                                                  entity_data[metric],
                                                  label=f'E{entity_id} - {style["label"]}',
                                                  color=style['color'],
                                                  linestyle=style['linestyle'],
                                                  alpha=0.7,
                                                  marker='o',
                                                  markersize=4)
                                all_lines.extend(line)
                                all_labels.append(f'E{entity_id} - {style["label"]}')

                # Set labels and title
                ax_ts.set_xlabel('Time')
                ax_ts.set_ylabel('CPU Utilization (%)')
                if ax_memory is not None:
                    ax_memory.set_ylabel('Memory Utilization (%)')
                if ax_network is not None:
                    ax_network.set_ylabel('Network Traffic (KB/s)')
                ax_ts.set_title("Metrics Over Time for Anomalous Entities", fontsize=12)

                # Add legend
                if all_lines and all_labels:
                    ax_ts.legend(all_lines, all_labels, 
                             loc='center left',
                             bbox_to_anchor=(1.25, 0.5))

                # Grid and formatting
                ax_ts.grid(True, alpha=0.3)
                plt.setp(ax_ts.get_xticklabels(), rotation=45, ha='right')

            # 3. Pattern Distribution (Bottom Right)
            ax_dist = fig.add_subplot(gs[1, 1])
            if anomalies:
                pattern_counts = pd.Series(anomalies.values()).value_counts()
                colors = [pattern_colors.get(p, 'gray') for p in pattern_counts.index]
                pattern_counts.plot(kind='bar', ax=ax_dist, color=colors)
                ax_dist.set_title("Anomaly Pattern Distribution", fontsize=12)
                ax_dist.set_xlabel("Pattern Type")
                ax_dist.set_ylabel("Count")
                plt.setp(ax_dist.get_xticklabels(), rotation=45, ha='right')

            # Set overall title and adjust layout
            if title:
                fig.suptitle(title, y=1.02, fontsize=16, fontweight='bold')
            fig.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Error in plot_combined_analysis: {str(e)}")
            return None