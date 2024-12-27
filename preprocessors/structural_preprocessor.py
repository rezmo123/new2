import logging
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import networkx as nx  # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder  # type: ignore
from typing import Tuple, Any, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StructuralPreprocessor:
    def __init__(self):
        """Initialize StructuralPreprocessor with enhanced feature extraction"""
        self.node_scaler = StandardScaler()
        self.edge_scaler = StandardScaler()
        self.service_encoder = LabelEncoder()
        self.n_node_features: Optional[int] = None
        self.n_edge_features: Optional[int] = None

    def extract_topological_features(self, G: nx.DiGraph) -> Tuple[np.ndarray, np.ndarray]:
        """Extract advanced topological features from graph structure"""
        try:
            # Node-level features
            node_features = []
            nodes = sorted(G.nodes())  # Ensure consistent ordering

            # Pre-calculate centrality metrics to avoid repeated computation
            betweenness_centrality = nx.betweenness_centrality(G)
            closeness_centrality = nx.closeness_centrality(G)
            pagerank = nx.pagerank(G)
            clustering_coeffs = nx.clustering(G)

            for node in nodes:
                # Basic centrality metrics
                degree = G.degree(node)
                in_degree = G.in_degree(node)
                out_degree = G.out_degree(node)
                clustering_coeff = clustering_coeffs[node]

                # Advanced centrality metrics
                betweenness = betweenness_centrality[node]
                closeness = closeness_centrality[node]
                pr_value = pagerank[node]

                # Local structure metrics
                neighbors = list(G.neighbors(node))
                avg_neighbor_degree = np.mean([G.degree(n) for n in neighbors]) if neighbors else 0

                node_features.append([
                    degree, in_degree, out_degree, clustering_coeff,
                    betweenness, closeness, pr_value,
                    avg_neighbor_degree
                ])

            # Edge-level features
            edge_features = []
            edges = sorted(G.edges())  # Ensure consistent ordering
            edge_betweenness = nx.edge_betweenness_centrality(G)

            for edge in edges:
                source, target = edge
                # Structural similarity
                source_neighbors = set(G[source])
                target_neighbors = set(G[target])
                jaccard = len(source_neighbors & target_neighbors) / len(source_neighbors | target_neighbors) if (source_neighbors or target_neighbors) else 0

                # Node importance
                source_cent = closeness_centrality[source]
                target_cent = closeness_centrality[target]

                # Edge importance
                edge_between = edge_betweenness[edge]

                edge_features.append([
                    jaccard, source_cent, target_cent, edge_between
                ])

            return np.array(node_features), np.array(edge_features)

        except Exception as e:
            logger.error(f"Error extracting topological features: {str(e)}")
            raise

    def analyze_graph_patterns(self, G: nx.DiGraph) -> np.ndarray:
        """Analyze common graph patterns and motifs"""
        try:
            pattern_features = []
            nodes = sorted(G.nodes())

            # Create undirected version for triangle counting
            G_undirected = G.to_undirected()
            triangles = nx.triangles(G_undirected)

            for node in nodes:
                # Analyze local patterns
                local_graph = nx.ego_graph(G, node, radius=1)
                local_undirected = local_graph.to_undirected()

                # Pattern metrics
                n_triangles = triangles[node]
                try:
                    if len(local_graph) > 1:  # Only calculate if there are at least 2 nodes
                        avg_path_length = nx.average_shortest_path_length(local_undirected)
                    else:
                        avg_path_length = 0
                except nx.NetworkXError:
                    avg_path_length = 0

                density = nx.density(local_graph)

                pattern_features.append([n_triangles, avg_path_length, density])

            return np.array(pattern_features)

        except Exception as e:
            logger.error(f"Error analyzing graph patterns: {str(e)}")
            raise

    def prepare_structural_data(self, relationships_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, nx.DiGraph]:
        """Prepare structural data with enhanced feature extraction"""
        try:
            logger.info("Starting structural data preparation...")
            logger.info(f"Input DataFrame columns: {relationships_df.columns.tolist()}")
            logger.info(f"First few rows of relationships: \n{relationships_df.head()}")

            # Validate input data
            required_columns = ['source', 'target']
            missing_columns = [col for col in required_columns if col not in relationships_df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Create directed graph from relationships DataFrame
            G = nx.DiGraph()

            # Add edges and service types
            service_types = []
            for _, row in relationships_df.iterrows():
                # Validate source and target values
                if pd.isna(row['source']) or pd.isna(row['target']):
                    logger.warning(f"Skipping row with null values: {row}")
                    continue

                G.add_edge(row['source'], row['target'])
                service_types.extend([row.get('service_type', 'unknown')])

            if len(G.nodes()) == 0:
                raise ValueError("No valid edges found in the relationships DataFrame")

            # Encode service types
            unique_services = np.unique(service_types)
            self.service_encoder.fit(unique_services)

            # Extract features with validation
            node_features, edge_features = self.extract_topological_features(G)
            if len(node_features) == 0 or len(edge_features) == 0:
                raise ValueError("Failed to extract features from the graph")

            pattern_features = self.analyze_graph_patterns(G)

            # Combine node and pattern features
            combined_node_features = np.hstack([node_features, pattern_features])

            # Set feature dimensions if not already set
            if self.n_node_features is None:
                self.n_node_features = combined_node_features.shape[1]
                self.n_edge_features = edge_features.shape[1]
                logger.info(f"Set feature dimensions - nodes: {self.n_node_features}, edges: {self.n_edge_features}")

            # Scale features
            if not hasattr(self.node_scaler, 'n_samples_seen_') or self.node_scaler.n_samples_seen_ is None:
                combined_node_features = self.node_scaler.fit_transform(combined_node_features)
                edge_features = self.edge_scaler.fit_transform(edge_features)
            else:
                combined_node_features = self.node_scaler.transform(combined_node_features)
                edge_features = self.edge_scaler.transform(edge_features)

            # Reshape features for model input (batch_size=1)
            combined_node_features = np.expand_dims(combined_node_features, axis=0)
            edge_features = np.expand_dims(edge_features, axis=0)

            logger.info(f"Graph summary - Nodes: {len(G.nodes())}, Edges: {len(G.edges())}")
            logger.info(f"Prepared node features shape: {combined_node_features.shape}")
            logger.info(f"Prepared edge features shape: {edge_features.shape}")
            logger.info("Structural data preparation completed successfully")

            return combined_node_features, edge_features, G

        except Exception as e:
            logger.error(f"Error preparing structural data: {str(e)}")
            logger.error(f"DataFrame info:\n{relationships_df.info()}")
            raise

    def save_state(self) -> Dict[str, Any]:
        """Return preprocessor state for saving"""
        return {
            'n_node_features': self.n_node_features,
            'n_edge_features': self.n_edge_features,
            'node_scaler_mean': self.node_scaler.mean_.tolist() if hasattr(self.node_scaler, 'mean_') else None,
            'node_scaler_scale': self.node_scaler.scale_.tolist() if hasattr(self.node_scaler, 'scale_') else None,
            'edge_scaler_mean': self.edge_scaler.mean_.tolist() if hasattr(self.edge_scaler, 'mean_') else None,
            'edge_scaler_scale': self.edge_scaler.scale_.tolist() if hasattr(self.edge_scaler, 'scale_') else None,
            'service_encoder_classes': self.service_encoder.classes_.tolist() if hasattr(self.service_encoder, 'classes_') else None
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load preprocessor state"""
        self.n_node_features = state['n_node_features']
        self.n_edge_features = state['n_edge_features']

        # Restore scalers if they were fitted
        if state['node_scaler_mean'] is not None:
            self.node_scaler.mean_ = np.array(state['node_scaler_mean'])
            self.node_scaler.scale_ = np.array(state['node_scaler_scale'])
            self.node_scaler.n_samples_seen_ = 1

        if state['edge_scaler_mean'] is not None:
            self.edge_scaler.mean_ = np.array(state['edge_scaler_mean'])
            self.edge_scaler.scale_ = np.array(state['edge_scaler_scale'])
            self.edge_scaler.n_samples_seen_ = 1

        if state['service_encoder_classes'] is not None:
            self.service_encoder.classes_ = np.array(state['service_encoder_classes'])