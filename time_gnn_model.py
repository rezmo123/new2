import logging
import os
import json
import numpy as np  # type: ignore
import tensorflow as tf
from tensorflow import keras
import pandas as pd  # type: ignore
import networkx as nx  # type: ignore
from sklearn.preprocessing import LabelEncoder  # type: ignore
from sklearn.metrics import roc_curve, auc, precision_recall_curve  # type: ignore
from preprocessors.time_series_preprocessor import TimeSeriesPreprocessor
import matplotlib.pyplot as plt  # type: ignore
from typing import Any, Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Force TensorFlow to use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.set_visible_devices([], 'GPU')

class TimeGNNAnomalyDetector:
    def __init__(self, input_shape, n_categories, sequence_length=5):
        """Initialize TimeGNN Anomaly Detector with proper state management"""
        self.sequence_length = sequence_length

        # Handle both tuple and integer input shapes
        if isinstance(input_shape, tuple):
            self.sequence_length, self.n_features = input_shape
        else:
            self.n_features = input_shape

        self.n_categories = n_categories
        self.hidden_dim = 32
        self.latent_dim = 16
        self.model = None
        self.reconstruction_threshold = None
        self.is_trained = False  # Explicitly initialize training state
        self.preprocessor = TimeSeriesPreprocessor(sequence_length=self.sequence_length)

        try:
            tf.config.run_functions_eagerly(True)
            self.model = self._build_model()
            logger.info(f"TimeGNN-Autoencoder model initialized with sequence_length={self.sequence_length}")
        except Exception as e:
            logger.error(f"Error in TimeGNN initialization: {str(e)}")
            raise

    def _validate_data(self, features_df: pd.DataFrame, relationships_df: pd.DataFrame) -> None:
        """Validate input data before processing"""
        if features_df is None or relationships_df is None:
            raise ValueError("features_df and relationships_df must not be None")

        if len(features_df) == 0 or len(relationships_df) == 0:
            raise ValueError("Input DataFrames cannot be empty")

        required_feature_cols = ['entity_id', 'timestamp', 'CPUUtilization', 
                             'MemoryUtilization', 'NetworkIn', 'NetworkOut']
        required_rel_cols = ['source', 'target', 'service_type']

        missing_feature_cols = [col for col in required_feature_cols 
                             if col not in features_df.columns]
        missing_rel_cols = [col for col in required_rel_cols 
                         if col not in relationships_df.columns]

        if missing_feature_cols:
            raise ValueError(f"Missing required columns in features_df: {missing_feature_cols}")
        if missing_rel_cols:
            raise ValueError(f"Missing required columns in relationships_df: {missing_rel_cols}")

        # Validate relationship consistency
        all_entities = pd.concat([
            relationships_df['source'],
            relationships_df['target']
        ]).unique()

        feature_entities = features_df['entity_id'].unique()
        missing_entities = [e for e in all_entities if e not in feature_entities]

        if missing_entities:
            logger.warning(f"Some entities in relationships are missing from features: {missing_entities}")

    def _create_entities_df(self, relationships_df: pd.DataFrame) -> pd.DataFrame:
        """Create entities DataFrame from relationships with improved mapping"""
        try:
            # Get unique entities from both source and target columns
            unique_sources = relationships_df['source'].unique()
            unique_targets = relationships_df['target'].unique()
            all_entities = np.unique(np.concatenate([unique_sources, unique_targets]))

            # Create base entities DataFrame
            entities_df = pd.DataFrame({'entity_id': all_entities})

            # Create a more robust service type mapping
            service_type_map = {}

            # First pass: map source entities
            source_service_types = relationships_df.groupby('source')['service_type'].agg(
                lambda x: x.value_counts().index[0]
            ).to_dict()
            service_type_map.update(source_service_types)

            # Second pass: map target entities that weren't sources
            target_mappings = relationships_df[~relationships_df['target'].isin(source_service_types.keys())]
            target_service_types = target_mappings.groupby('target')['service_type'].agg(
                lambda x: x.value_counts().index[0]
            ).to_dict()
            service_type_map.update(target_service_types)

            # Apply service type mapping with fallback
            entities_df['service_type'] = entities_df['entity_id'].map(
                lambda x: service_type_map.get(x, 'unknown')
            )

            logger.info(f"Created entities DataFrame with {len(entities_df)} entities")
            logger.info(f"Service types distribution: {entities_df['service_type'].value_counts().to_dict()}")

            return entities_df

        except Exception as e:
            logger.error(f"Error creating entities DataFrame: {str(e)}")
            raise

    def train(self, timeseries_df: pd.DataFrame, relationships_df: pd.DataFrame, epochs: int = 10, threshold_percentile: float = 95) -> Any:
        """Train the model with proper state management and relationships data"""
        try:
            logger.info("Starting TimeGNN-Autoencoder model training...")
            self.is_trained = False  # Reset training state at start

            # Validate input data
            self._validate_data(timeseries_df, relationships_df)

            # Create entities DataFrame from relationships
            entities_df = self._create_entities_df(relationships_df)

            # Prepare sequences using preprocessor
            sequences, categories = self.preprocessor.prepare_temporal_data(
                timeseries_df,
                entities_df
            )

            if len(sequences) == 0:
                raise ValueError("No valid sequences could be generated")

            # Train model
            history = self.model.fit(
                [sequences, categories],  # Input both features and categories
                sequences,  # Autoencoder reconstructs input
                epochs=epochs,
                batch_size=16,
                validation_split=0.2,
                verbose=1
            )

            # Calculate reconstruction errors and threshold
            predictions = self.model.predict([sequences, categories])
            reconstruction_errors = np.mean(np.square(sequences - predictions), axis=(1, 2))
            self.reconstruction_threshold = np.percentile(reconstruction_errors, threshold_percentile)

            # Validate anomaly detection
            anomalies = reconstruction_errors > self.reconstruction_threshold
            anomaly_rate = np.mean(anomalies)

            if anomaly_rate == 0:
                logger.warning("No anomalies detected in training data. Adjusting threshold...")
                self.reconstruction_threshold *= 0.8
            elif anomaly_rate > 0.3:
                logger.warning("Too many anomalies detected. Adjusting threshold...")
                self.reconstruction_threshold *= 1.2

            # Visualize training history if available
            if history and hasattr(history, 'history'):
                self._visualize_training_history(history)

            self.is_trained = True
            logger.info(f"Training completed. Final loss: {history.history['loss'][-1]:.4f}")
            logger.info(f"Anomaly detection rate: {anomaly_rate:.2%}")

            return history

        except Exception as e:
            self.is_trained = False
            logger.error(f"Error during training: {str(e)}")
            raise

    def predict(self, features_df: pd.DataFrame, relationships_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Detect temporal anomalies with proper state validation and relationships data"""
        try:
            # Validate model state
            if not self.is_trained:
                raise ValueError("Model must be trained before prediction")

            if self.reconstruction_threshold is None:
                raise ValueError("Reconstruction threshold not set")

            # Validate input data using the same validation as training
            self._validate_data(features_df, relationships_df)

            # Create entities DataFrame from relationships
            entities_df = self._create_entities_df(relationships_df)

            # Prepare sequences using the same preprocessor
            sequences, categories = self.preprocessor.prepare_temporal_data(
                features_df,
                entities_df
            )

            if len(sequences) == 0:
                raise ValueError("No valid sequences could be generated")

            # Get predictions
            predictions = self.model.predict([sequences, categories])
            reconstruction_errors = np.mean(np.square(sequences - predictions), axis=(1, 2))

            # Detect anomalies
            anomalies = reconstruction_errors > self.reconstruction_threshold

            # Classify patterns
            pattern_types = self._classify_patterns(reconstruction_errors, self.reconstruction_threshold)

            logger.info(f"Detection completed. Found {np.sum(anomalies)} anomalies")
            logger.info(f"Anomaly rate: {(np.sum(anomalies)/len(anomalies)):.2%}")

            return anomalies, pattern_types

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def _classify_patterns(self, reconstruction_errors: np.ndarray, threshold: float) -> np.ndarray:
        """Classify anomaly patterns with more granular thresholds"""
        pattern_types = np.full_like(reconstruction_errors, fill_value='normal', dtype=object)
        anomalies = reconstruction_errors > threshold

        # Convert boolean mask to integer indices
        anomaly_indices = np.where(anomalies)[0]

        for idx in anomaly_indices:
            error_magnitude = reconstruction_errors[idx] / threshold
            if error_magnitude > 2.5:
                pattern_types[idx] = 'spike'
            elif error_magnitude > 1.75:
                pattern_types[idx] = 'drop'
            elif error_magnitude > 1.25:
                pattern_types[idx] = 'gradual'
            else:
                pattern_types[idx] = 'pattern'

        return pattern_types

    def save_model(self, path: str) -> bool:
        """Save the TimeGNN-Autoencoder model with all parameters"""
        try:
            if not self.is_trained:
                raise ValueError("Cannot save untrained model. Train the model first.")

            os.makedirs(path, exist_ok=True)
            model_path = os.path.join(path, 'keras_model')
            self.model.save(model_path)

            # Save preprocessor state and model properties
            properties = {
                'n_features': self.n_features,
                'n_categories': self.n_categories,
                'reconstruction_threshold': float(self.reconstruction_threshold) 
                    if self.reconstruction_threshold is not None else None,
                'hidden_dim': self.hidden_dim,
                'latent_dim': self.latent_dim,
                'is_trained': self.is_trained,
                'preprocessor_state': self.preprocessor.save_state()
            }

            with open(os.path.join(path, 'model_properties.json'), 'w') as f:
                json.dump(properties, f, indent=4)

            logger.info(f"Model and properties saved to {path}")
            return True

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, path: str) -> bool:
        """Load the TimeGNN-Autoencoder model with all parameters"""
        try:
            if not os.path.exists(path):
                raise ValueError(f"Model path {path} does not exist")

            # Load the keras model
            model_path = os.path.join(path, 'keras_model')
            if not os.path.exists(model_path):
                raise ValueError(f"Keras model not found in {path}")

            self.model = keras.models.load_model(model_path)

            # Load properties
            properties_path = os.path.join(path, 'model_properties.json')
            if not os.path.exists(properties_path):
                raise ValueError(f"Model properties not found in {path}")

            with open(properties_path, 'r') as f:
                properties = json.load(f)

            # Restore properties
            self.n_features = properties['n_features']
            self.n_categories = properties['n_categories']
            self.reconstruction_threshold = properties['reconstruction_threshold']
            self.hidden_dim = properties['hidden_dim']
            self.latent_dim = properties['latent_dim']
            self.is_trained = properties.get('is_trained', False)  # Load training state with default False

            # Load preprocessor state
            self.preprocessor.load_state(properties['preprocessor_state'])

            logger.info(f"Model and properties loaded from {path}")
            logger.info(f"Model training state: {'trained' if self.is_trained else 'untrained'}")
            if self.reconstruction_threshold:
                logger.info(f"Current reconstruction threshold: {self.reconstruction_threshold:.3f}")
            return True

        except Exception as e:
            self.is_trained = False  # Reset training state on load failure
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _build_model(self) -> keras.Model:
        """Build TimeGNN-Autoencoder model with proper input handling"""
        try:
            # Input layers with explicit shape
            node_features = keras.layers.Input(shape=(self.sequence_length, self.n_features))
            category_input = keras.layers.Input(shape=(1,), dtype='int32')

            # Category embedding
            category_embedding = keras.layers.Embedding(
                input_dim=self.n_categories,
                output_dim=self.hidden_dim
            )(category_input)
            category_embedding = keras.layers.Flatten()(category_embedding)
            category_projection = keras.layers.Dense(self.hidden_dim)(category_embedding)
            category_projection = keras.layers.RepeatVector(self.sequence_length)(category_projection)

            # Encoder path
            x = keras.layers.Dense(self.hidden_dim)(node_features)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('relu')(x)

            # Temporal attention
            attention = keras.layers.MultiHeadAttention(
                num_heads=4,
                key_dim=self.hidden_dim
            )(x, x)
            x = keras.layers.Add()([x, attention])
            x = keras.layers.LayerNormalization()(x)

            # Bidirectional LSTM encoder
            lstm_out = keras.layers.Bidirectional(
                keras.layers.LSTM(
                    self.hidden_dim,
                    return_sequences=True,
                    recurrent_dropout=0.1
                )
            )(x)

            # Combine with category information
            combined = keras.layers.Concatenate()([lstm_out, category_projection])

            # Bottleneck layer (compressed representation)
            bottleneck = keras.layers.Dense(self.latent_dim)(combined)
            bottleneck = keras.layers.BatchNormalization()(bottleneck)
            bottleneck = keras.layers.Activation('relu')(bottleneck)

            # Decoder path with proper dimensionality
            x = keras.layers.Dense(self.hidden_dim)(bottleneck)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('relu')(x)

            # Decoder LSTM
            x = keras.layers.Bidirectional(
                keras.layers.LSTM(
                    self.hidden_dim,
                    return_sequences=True,
                    recurrent_dropout=0.1
                )
            )(x)

            # Final reconstruction
            output = keras.layers.TimeDistributed(
                keras.layers.Dense(self.n_features)
            )(x)

            # Create model with both inputs
            model = keras.Model(
                inputs=[node_features, category_input],
                outputs=output
            )

            # Use legacy optimizer for better compatibility
            optimizer = keras.optimizers.legacy.Adam(learning_rate=0.001)
            model.compile(optimizer=optimizer, loss=keras.losses.Huber())

            self.is_trained = False  # Reset training state when building new model
            return model

        except Exception as e:
            logger.error(f"Error building TimeGNN-Autoencoder model: {str(e)}")
            raise
    
    def _visualize_training_history(self, history: Any) -> None:
        """Visualize training metrics with proper validation"""
        try:
            if not history or not hasattr(history, 'history') or not history.history:
                logger.warning("No training history available for visualization")
                return

            # Ensure we have valid data before attempting to plot
            if 'loss' not in history.history or len(history.history['loss']) == 0:
                logger.warning("No loss data found in training history")
                return

            plt.figure(figsize=(10, 6))

            # Only plot metrics that exist and have data
            for metric in ['loss', 'val_loss']:
                if metric in history.history and len(history.history[metric]) > 0:
                    values = history.history[metric]
                    epochs = range(len(values))  # Use actual length of values
                    plt.plot(epochs, values, label=metric, marker='o')

            plt.title('Model Training History')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

            # Save the plot
            os.makedirs('visualizations', exist_ok=True)
            plt.savefig(os.path.join('visualizations', 'training_history.png'))
            plt.close()

        except Exception as e:
            logger.error(f"Error visualizing training history: {str(e)}")
            logger.error("Continuing despite visualization error")