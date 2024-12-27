import logging
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import networkx as nx
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, List, Optional, Tuple, Union
import time
from ..preprocessors.structural_preprocessor import StructuralPreprocessor
import pandas as pd

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Force TensorFlow to use CPU for consistent production deployment
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.set_visible_devices([], 'GPU')

class StructuralGNNDetector:
    """Production-ready Structural Graph Neural Network for anomaly detection.

    This implementation includes robust error handling, comprehensive logging,
    and performance monitoring for production deployment.
    """

    def __init__(self, n_node_features: Optional[int] = None, 
                 n_edge_features: Optional[int] = None,
                 hidden_dim: int = 64,
                 learning_rate: float = 0.001):
        """Initialize Structural GNN with enhanced configuration options.

        Args:
            n_node_features: Number of node features
            n_edge_features: Number of edge features
            hidden_dim: Dimension of hidden layers
            learning_rate: Learning rate for model optimization
        """
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.model = None
        self.reconstruction_threshold = None
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.is_trained = False
        self.training_history: Dict[str, List[float]] = {
            'loss': [],
            'training_time': []
        }
        self.preprocessor = StructuralPreprocessor()

        try:
            tf.config.run_functions_eagerly(True)
            logger.info(
                f"Initialized StructuralGNN with config: hidden_dim={hidden_dim}, "
                f"learning_rate={learning_rate}"
            )
        except Exception as e:
            logger.error(f"Error in StructuralGNN initialization: {str(e)}")
            raise

    def _validate_model_state(self) -> None:
        """Validate model state before operations."""
        if self.model is None:
            raise ValueError("Model not initialized. Call build_model() first.")

    def _validate_feature_dimensions(self, node_features: np.ndarray, 
                                   edge_features: np.ndarray) -> None:
        """Validate input feature dimensions."""
        if node_features.ndim != 3:
            raise ValueError(f"Expected 3D node features, got shape {node_features.shape}")
        if edge_features.ndim != 3:
            raise ValueError(f"Expected 3D edge features, got shape {edge_features.shape}")

    def _build_model(self) -> None:
        """Build GNN model with enhanced architecture and monitoring."""
        try:
            if self.n_node_features is None or self.n_edge_features is None:
                raise ValueError("Feature dimensions must be set before building model")

            # Input layers with explicit batch dimension
            node_features = keras.layers.Input(
                shape=(None, self.n_node_features), 
                name='node_features'
            )
            edge_features = keras.layers.Input(
                shape=(None, self.n_edge_features), 
                name='edge_features'
            )
            adjacency_matrix = keras.layers.Input(
                shape=(None, None), 
                sparse=True, 
                name='adjacency'
            )

            # Enhanced feature processing with residual connections
            x = node_features
            edge_x = edge_features

            # Graph convolution blocks with improved architecture
            for i in range(3):
                # Node feature processing with residual connections
                node_residual = x
                x = keras.layers.Dense(self.hidden_dim, name=f'node_dense_{i}')(x)
                x = keras.layers.BatchNormalization(name=f'node_bn_{i}')(x)
                x = keras.layers.Activation('relu', name=f'node_relu_{i}')(x)

                # Multi-head self-attention for nodes
                node_attention = keras.layers.MultiHeadAttention(
                    num_heads=4,
                    key_dim=self.hidden_dim // 4,
                    name=f'node_attention_{i}'
                )(x, x)
                x = keras.layers.Add(name=f'node_residual_{i}')([x, node_attention])
                x = keras.layers.LayerNormalization(name=f'node_norm_{i}')(x)

                # Add residual connection
                if i > 0:  # Skip first layer to match dimensions
                    x = keras.layers.Add(name=f'node_skip_{i}')([x, node_residual])

                # Edge feature processing with similar architecture
                edge_residual = edge_x
                edge_x = keras.layers.Dense(self.hidden_dim, name=f'edge_dense_{i}')(edge_x)
                edge_x = keras.layers.BatchNormalization(name=f'edge_bn_{i}')(edge_x)
                edge_x = keras.layers.Activation('relu', name=f'edge_relu_{i}')(edge_x)

                # Multi-head self-attention for edges
                edge_attention = keras.layers.MultiHeadAttention(
                    num_heads=4,
                    key_dim=self.hidden_dim // 4,
                    name=f'edge_attention_{i}'
                )(edge_x, edge_x)
                edge_x = keras.layers.Add(name=f'edge_residual_{i}')([edge_x, edge_attention])
                edge_x = keras.layers.LayerNormalization(name=f'edge_norm_{i}')(edge_x)

                # Add residual connection for edges
                if i > 0:
                    edge_x = keras.layers.Add(name=f'edge_skip_{i}')([edge_x, edge_residual])

            # Improved bottleneck with separate node and edge encodings
            node_encoded = keras.layers.Dense(self.hidden_dim // 2, name='node_bottleneck')(x)
            edge_encoded = keras.layers.Dense(self.hidden_dim // 2, name='edge_bottleneck')(edge_x)

            # Enhanced decoder architecture
            x = keras.layers.Dense(self.hidden_dim, name='node_decoder_1')(node_encoded)
            x = keras.layers.BatchNormalization(name='node_decoder_bn')(x)
            x = keras.layers.Activation('relu', name='node_decoder_relu')(x)
            node_output = keras.layers.Dense(self.n_node_features, name='node_output')(x)

            edge_x = keras.layers.Dense(self.hidden_dim, name='edge_decoder_1')(edge_encoded)
            edge_x = keras.layers.BatchNormalization(name='edge_decoder_bn')(edge_x)
            edge_x = keras.layers.Activation('relu', name='edge_decoder_relu')(edge_x)
            edge_output = keras.layers.Dense(self.n_edge_features, name='edge_output')(edge_x)

            # Create model with enhanced input handling
            self.model = keras.Model(
                inputs={
                    'node_features': node_features,
                    'edge_features': edge_features,
                    'adjacency': adjacency_matrix
                },
                outputs={
                    'node_output': node_output,
                    'edge_output': edge_output
                }
            )

            # Compile with improved optimizer configuration
            optimizer = keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                clipnorm=1.0  # Gradient clipping for stability
            )

            self.model.compile(
                optimizer=optimizer,
                loss={
                    'node_output': keras.losses.Huber(delta=1.0),
                    'edge_output': keras.losses.Huber(delta=1.0)
                },
                loss_weights={
                    'node_output': 1.0,
                    'edge_output': 1.0
                },
                metrics=['mae']  # Add mean absolute error for monitoring
            )

            logger.info(f"Model architecture built successfully")
            self.model.summary(print_fn=logger.info)

        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise

    def train(self, relationships_df: pd.DataFrame, 
             epochs: int = 10, 
             threshold_percentile: float = 95,
             batch_size: int = 32) -> Dict[str, Any]:
        """Train the model with enhanced monitoring.
        Args:
            relationships_df: DataFrame containing entity relationships
            epochs: Number of training epochs
            threshold_percentile: Percentile for anomaly threshold
            batch_size: Training batch size
        Returns:
            Dictionary containing training metrics and history
        """
        try:
            start_time = time.time()
            logger.info(f"Starting model training with {epochs} epochs")

            # Prepare data with enhanced preprocessing
            node_features, edge_features, G = self.preprocessor.prepare_structural_data(
                relationships_df
            )

            self._validate_feature_dimensions(node_features, edge_features)

            # Initialize model if needed
            if self.model is None:
                self.n_node_features = node_features.shape[-1]
                self.n_edge_features = edge_features.shape[-1]
                self._build_model()

            # Create sparse adjacency matrix
            adj_matrix = nx.adjacency_matrix(G)
            indices = np.array([[i, j] for i, j in zip(*adj_matrix.nonzero())], 
                             dtype=np.int64)
            values = adj_matrix.data.astype(np.float32)
            shape = adj_matrix.shape

            adj_sparse = tf.sparse.SparseTensor(
                indices=indices,
                values=values,
                dense_shape=shape
            )

            # Train with comprehensive monitoring
            history = self.model.fit(
                {
                    'node_features': node_features,
                    'edge_features': edge_features,
                    'adjacency': adj_sparse
                },
                {
                    'node_output': node_features,
                    'edge_output': edge_features
                },
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )

            # Calculate reconstruction errors and threshold
            predictions = self.model.predict({
                'node_features': node_features,
                'edge_features': edge_features,
                'adjacency': adj_sparse
            })

            node_errors = np.mean(np.square(node_features - predictions['node_output']), 
                                axis=(1, 2))
            edge_errors = np.mean(np.square(edge_features - predictions['edge_output']), 
                                axis=(1, 2))
            reconstruction_errors = node_errors + edge_errors

            self.reconstruction_threshold = np.percentile(
                reconstruction_errors, 
                threshold_percentile
            )

            # Update training history
            training_time = time.time() - start_time
            self.training_history['loss'].extend(history.history['loss'])
            self.training_history['training_time'].append(training_time)

            self.is_trained = True

            logger.info(f"Training completed in {training_time:.2f} seconds")
            logger.info(f"Final loss: {history.history['loss'][-1]:.4f}")
            logger.info(f"Set reconstruction threshold: {self.reconstruction_threshold:.4f}")

            return {
                'history': history.history,
                'training_time': training_time,
                'final_loss': history.history['loss'][-1],
                'reconstruction_threshold': self.reconstruction_threshold
            }

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def predict(self, relationships_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect structural anomalies with comprehensive pattern analysis.

        Args:
            relationships_df: DataFrame containing entity relationships

        Returns:
            Tuple containing anomaly flags, pattern types, and reconstruction errors
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before prediction")

            self._validate_model_state()

            logger.info("Starting anomaly detection")
            prediction_start = time.time()

            # Prepare data
            node_features, edge_features, G = self.preprocessor.prepare_structural_data(
                relationships_df
            )

            self._validate_feature_dimensions(node_features, edge_features)

            # Create sparse adjacency matrix
            adj_matrix = nx.adjacency_matrix(G)
            indices = np.array([[i, j] for i, j in zip(*adj_matrix.nonzero())], 
                             dtype=np.int64)
            values = adj_matrix.data.astype(np.float32)
            shape = adj_matrix.shape

            adj_sparse = tf.sparse.SparseTensor(
                indices=indices,
                values=values,
                dense_shape=shape
            )

            # Get predictions
            predictions = self.model.predict({
                'node_features': node_features,
                'edge_features': edge_features,
                'adjacency': adj_sparse
            })

            # Calculate reconstruction errors
            node_errors = np.mean(np.square(node_features - predictions['node_output']), 
                                axis=(1, 2))
            edge_errors = np.mean(np.square(edge_features - predictions['edge_output']), 
                                axis=(1, 2))
            reconstruction_errors = node_errors + edge_errors

            # Detect anomalies
            anomalies = reconstruction_errors > self.reconstruction_threshold

            # Enhanced pattern analysis
            pattern_types = np.full_like(anomalies, fill_value='normal', dtype='object')
            edges = list(G.edges())

            for i in range(len(reconstruction_errors)):
                if anomalies[i]:
                    # Calculate relative contributions
                    node_contribution = node_errors[i] / reconstruction_errors[i]
                    edge_contribution = edge_errors[i] / reconstruction_errors[i]

                    # Advanced metrics
                    node_complexity = np.std(node_features[i])
                    edge_density = (len(edges) / (len(G.nodes) * (len(G.nodes) - 1))
                                  if len(G.nodes) > 1 else 0)

                    # Comprehensive pattern classification
                    if node_contribution > 0.7:
                        if node_complexity > 0.5:
                            pattern_types[i] = 'complex_node_structural'
                        else:
                            pattern_types[i] = 'simple_node_structural'
                    elif edge_contribution > 0.7:
                        if edge_density > 0.5:
                            pattern_types[i] = 'dense_edge_structural'
                        else:
                            pattern_types[i] = 'sparse_edge_structural'
                    else:
                        pattern_types[i] = 'hybrid_structural'

            prediction_time = time.time() - prediction_start
            n_anomalies = int(np.sum(anomalies))

            logger.info(f"Detection completed in {prediction_time:.2f} seconds")
            logger.info(f"Found {n_anomalies} anomalies "
                       f"({(n_anomalies/len(anomalies))*100:.2f}% of total)")

            if n_anomalies > 0:
                pattern_counts = np.unique(pattern_types[anomalies], return_counts=True)
                logger.info("Pattern distribution:")
                for pattern, count in zip(*pattern_counts):
                    logger.info(f"- {pattern}: {count}")

            return anomalies.flatten(), pattern_types.flatten(), reconstruction_errors.flatten()

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def save_model(self, path: str) -> bool:
        """Save model with enhanced state preservation.

        Args:
            path: Path to save the model
        """
        try:
            if not self.is_trained:
                raise ValueError("Cannot save untrained model")

            self._validate_model_state()

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Save Keras model
            self.model.save(path)

            # Save additional properties
            properties = {
                'n_node_features': self.n_node_features,
                'n_edge_features': self.n_edge_features,
                'hidden_dim': self.hidden_dim,
                'learning_rate': self.learning_rate,
                'reconstruction_threshold': float(self.reconstruction_threshold),
                'is_trained': self.is_trained,
                'training_history': self.training_history,
                'preprocessor_state': self.preprocessor.save_state()
            }

            # Save properties
            properties_path = f"{path}_properties.json"
            with open(properties_path, 'w') as f:
                json.dump(properties, f, indent=4)

            logger.info(f"Model and properties saved to {path}")
            return True

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, path: str) -> bool:
        """Load model with comprehensive state restoration.

        Args:
            path: Path to load the model from
        """
        try:
            if not os.path.exists(path):
                raise ValueError(f"Model path {path} does not exist")

            # Load Keras model
            self.model = keras.models.load_model(path)

            # Load properties
            properties_path = f"{path}_properties.json"
            if not os.path.exists(properties_path):
                raise ValueError(f"Properties file not found at {properties_path}")

            with open(properties_path, 'r') as f:
                properties = json.load(f)

            # Restore properties
            self.n_node_features = properties['n_node_features']
            self.n_edge_features = properties['n_edge_features']
            self.hidden_dim = properties['hidden_dim']
            self.learning_rate = properties['learning_rate']
            self.reconstruction_threshold = properties['reconstruction_threshold']
            self.is_trained = properties['is_trained']
            self.training_history = properties['training_history']

            # Load preprocessor state
            self.preprocessor.load_state(properties['preprocessor_state'])

            logger.info(f"Model and properties loaded from {path}")
            logger.info(f"Model training state: {'trained' if self.is_trained else 'untrained'}")

            return True

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise