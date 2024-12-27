import logging
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import networkx as nx
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Force TensorFlow to use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.set_visible_devices([], 'GPU')

class GNNAnomalyDetector:
    def __init__(self, n_features, n_categories):
        """Initialize GNN model for unsupervised anomaly detection"""
        self.n_features = n_features
        self.n_categories = n_categories
        self.hidden_dim = 32
        self.model = None
        self.reconstruction_threshold = None
        self.service_type_map = None
        self.categorical_encoders = {}
        self.categorical_features = []
        self.numerical_features = []
        self.feature_means = None
        self.feature_stds = None

        try:
            tf.config.run_functions_eagerly(True)
            self.model = self._build_model()
            logger.info("GNN model initialized successfully")
        except Exception as e:
            logger.error(f"Error in GNN initialization: {str(e)}")
            raise

    def identify_categorical_features(self, df):
        """Identify categorical features, excluding anomaly-related columns"""
        categorical_features = []
        numerical_features = []

        for column in df.columns:
            # Skip metadata and anomaly-related columns
            if column in ['entity_id', 'timestamp', 'is_anomaly', 'anomaly_type']:
                continue

            # Check if column is explicitly categorical
            if df[column].dtype == 'object' or df[column].dtype == 'category':
                categorical_features.append(column)
                continue

            # Check if numerical column is actually categorical
            unique_values = df[column].nunique()
            if unique_values < 10 and (df[column] % 1 == 0).all():
                categorical_features.append(column)
            else:
                numerical_features.append(column)

        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        logger.info(f"Identified categorical features: {categorical_features}")
        logger.info(f"Identified numerical features: {numerical_features}")

        return categorical_features, numerical_features

    def encode_categorical_features(self, df):
        """Encode categorical features using LabelEncoder"""
        encoded_data = df.copy()

        for feature in self.categorical_features:
            if feature not in self.categorical_encoders:
                self.categorical_encoders[feature] = LabelEncoder()
                encoded_data[feature] = self.categorical_encoders[feature].fit_transform(df[feature])
            else:
                encoded_data[feature] = self.categorical_encoders[feature].transform(df[feature])

        return encoded_data

    def _build_model(self):
        """Build GNN model for unsupervised anomaly detection"""
        try:
            # Input layers
            node_features = keras.layers.Input(shape=(self.n_features,))
            category_input = keras.layers.Input(shape=(1,), dtype='int32')
            adjacency_matrix = keras.layers.Input(shape=(None,), sparse=True)

            # Category embedding
            category_embedding = keras.layers.Embedding(
                input_dim=self.n_categories,
                output_dim=self.hidden_dim
            )(category_input)
            category_embedding = keras.layers.Flatten()(category_embedding)
            category_projection = keras.layers.Dense(self.hidden_dim)(category_embedding)

            # Encoder
            x = keras.layers.Dense(self.hidden_dim)(node_features)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('relu')(x)

            # Graph convolution layers
            for _ in range(2):
                x = keras.layers.Dense(self.hidden_dim)(x)
                x = keras.layers.BatchNormalization()(x)
                x = keras.layers.Dropout(0.2)(x)

            # Bottleneck layer
            bottleneck = keras.layers.Dense(self.hidden_dim // 2)(x)

            # Decoder
            x = keras.layers.Dense(self.hidden_dim)(bottleneck)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('relu')(x)

            # Combine with category information
            combined = keras.layers.Concatenate()([x, category_projection])
            combined = keras.layers.Dense(self.hidden_dim)(combined)
            combined = keras.layers.BatchNormalization()(combined)

            # Output reconstruction
            output = keras.layers.Dense(self.n_features)(combined)

            # Create model
            model = keras.Model(
                inputs=[node_features, category_input, adjacency_matrix],
                outputs=output
            )

            # Use Huber loss for robust reconstruction
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss=keras.losses.Huber()
            )

            return model

        except Exception as e:
            logger.error(f"Error building GNN model: {str(e)}")
            raise

    def prepare_graph_data(self, features_df, relationships_df):
        """Prepare graph data for unsupervised learning"""
        try:
            # Create graph
            G = nx.from_pandas_edgelist(relationships_df, 'source', 'target')
            adj_matrix = nx.adjacency_matrix(G)

            # Create sparse tensor
            indices = np.array([[i, j] for i, j in zip(*adj_matrix.nonzero())], dtype=np.int64)
            values = adj_matrix.data.astype(np.float32)
            shape = adj_matrix.shape

            adj_sparse = tf.sparse.SparseTensor(
                indices=indices,
                values=values,
                dense_shape=shape
            )

            # Process features
            self.identify_categorical_features(features_df)
            encoded_df = self.encode_categorical_features(features_df)

            # Extract numerical features
            features = encoded_df[self.numerical_features].values.astype(np.float32)

            # Normalize features
            if self.feature_means is None or self.feature_stds is None:
                self.feature_means = np.mean(features, axis=0)
                self.feature_stds = np.std(features, axis=0)
                self.feature_stds[self.feature_stds == 0] = 1

            features = (features - self.feature_means) / self.feature_stds

            # Create service type mapping
            unique_service_types = relationships_df['service_type'].unique()
            if self.service_type_map is None:
                self.service_type_map = {st: i for i, st in enumerate(unique_service_types)}

            # Map service types for each entity
            entity_service_types = {}
            for _, row in relationships_df.iterrows():
                entity_service_types[row['source']] = row['service_type']
                entity_service_types[row['target']] = row['service_type']

            # Convert to category indices
            categories = np.array([
                self.service_type_map[entity_service_types[entity_id]]
                for entity_id in features_df['entity_id']
            ])
            categories = categories.reshape(-1, 1)

            return features, categories, adj_sparse, features.copy()

        except Exception as e:
            logger.error(f"Error preparing graph data: {str(e)}")
            raise

    def train(self, features_df, relationships_df, epochs=10):
        """Train the GNN model in unsupervised mode"""
        try:
            logger.info("Training GNN model...")
            features, categories, adj_sparse, reconstruction_targets = self.prepare_graph_data(
                features_df, relationships_df
            )

            # Train model to minimize reconstruction error
            history = self.model.fit(
                [features, categories, adj_sparse],
                reconstruction_targets,
                epochs=epochs,
                batch_size=32,
                verbose=1
            )

            # Calculate reconstruction errors for threshold
            predictions = self.model.predict([features, categories, adj_sparse])
            reconstruction_errors = np.mean(np.square(reconstruction_targets - predictions), axis=1)

            # Set threshold at 95th percentile of reconstruction errors
            self.reconstruction_threshold = np.percentile(reconstruction_errors, 95)

            logger.info(f"Training completed. Threshold: {self.reconstruction_threshold:.3f}")
            return history

        except Exception as e:
            logger.error(f"Error training GNN model: {str(e)}")
            raise

    def predict(self, features_df, relationships_df):
        """Detect anomalies based on reconstruction error"""
        try:
            features, categories, adj_sparse, _ = self.prepare_graph_data(
                features_df, relationships_df
            )

            # Get reconstructions
            predictions = self.model.predict([features, categories, adj_sparse])

            # Calculate reconstruction errors
            reconstruction_errors = np.mean(np.square(features - predictions), axis=1)

            # Detect anomalies
            anomalies = reconstruction_errors > self.reconstruction_threshold

            # Analyze anomaly patterns
            pattern_types = np.full_like(anomalies, fill_value='normal', dtype='object')
            for i in range(len(reconstruction_errors)):
                if anomalies[i]:
                    error_magnitude = reconstruction_errors[i] / self.reconstruction_threshold
                    if error_magnitude > 3.0:
                        pattern_types[i] = 'spike'
                    elif error_magnitude > 2.0:
                        pattern_types[i] = 'drop'
                    else:
                        pattern_types[i] = 'gradual'

            logger.info(f"Detection threshold: {self.reconstruction_threshold:.3f}")
            logger.info(f"Anomalies detected: {int(np.sum(anomalies))}")

            return anomalies, pattern_types

        except Exception as e:
            logger.error(f"Error in GNN prediction: {str(e)}")
            raise

    def save_model(self, path):
        """Save model with all parameters"""
        try:
            # Save Keras model
            self.model.save(path)

            # Save additional properties
            properties = {
                'n_features': self.n_features,
                'n_categories': self.n_categories,
                'reconstruction_threshold': float(self.reconstruction_threshold) if self.reconstruction_threshold is not None else None,
                'service_type_map': self.service_type_map,
                'hidden_dim': self.hidden_dim,
                'categorical_features': self.categorical_features,
                'numerical_features': self.numerical_features,
                'categorical_encoders': {k: v.classes_.tolist() for k, v in self.categorical_encoders.items()},
                'feature_means': self.feature_means.tolist() if self.feature_means is not None else None,
                'feature_stds': self.feature_stds.tolist() if self.feature_stds is not None else None
            }

            # Save properties
            properties_path = f"{path}_properties.json"
            with open(properties_path, 'w') as f:
                import json
                json.dump(properties, f, indent=4)

            logger.info(f"Model saved successfully to {path}")
            return True

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, path):
        """Load model with all parameters"""
        try:
            # Load Keras model
            self.model = keras.models.load_model(path)

            # Load properties
            properties_path = f"{path}_properties.json"
            with open(properties_path, 'r') as f:
                import json
                properties = json.load(f)

            # Restore properties
            self.n_features = properties['n_features']
            self.n_categories = properties['n_categories']
            self.reconstruction_threshold = properties['reconstruction_threshold']
            self.service_type_map = properties['service_type_map']
            self.hidden_dim = properties['hidden_dim']
            self.categorical_features = properties['categorical_features']
            self.numerical_features = properties['numerical_features']
            self.feature_means = np.array(properties['feature_means']) if properties['feature_means'] is not None else None
            self.feature_stds = np.array(properties['feature_stds']) if properties['feature_stds'] is not None else None

            # Restore encoders
            for feature, classes in properties['categorical_encoders'].items():
                encoder = LabelEncoder()
                encoder.classes_ = np.array(classes)
                self.categorical_encoders[feature] = encoder

            logger.info(f"Model loaded successfully from {path}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise