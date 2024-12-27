"""TimeGNN Model Implementation
Contains the core temporal graph neural network model for anomaly detection.
"""

import logging
import os
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Force TensorFlow to use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.set_visible_devices([], 'GPU')

class TimeGNNAnomalyDetector:
    def __init__(self, input_shape: Tuple[int, int],
                 n_categories: int,
                 hidden_dim: int = 64,
                 learning_rate: float = 0.001):
        try:
            self.input_shape = input_shape
            self.n_categories = n_categories
            self.hidden_dim = hidden_dim
            self.learning_rate = learning_rate
            self.model = None
            self.category_encoder = LabelEncoder()
            self.threshold = None
            self.is_trained = False
            self.training_history = {
                'loss': [],
                'val_loss': [],
                'training_time': []
            }
            # Update numeric features to match available test data
            self.numeric_features = [
                'CPUUtilization', 'MemoryUtilization', 
                'NetworkIn', 'NetworkOut'
            ]

            logger.info(
                f"Initialized TimeGNN with config: input_shape={input_shape}, "
                f"n_categories={n_categories}, hidden_dim={hidden_dim}, "
                f"learning_rate={learning_rate}"
            )

        except Exception as e:
            logger.error(f"Error in TimeGNN initialization: {str(e)}")
            raise

    def _build_model(self) -> None:
        """Build TimeGNN model with enhanced architecture and monitoring."""
        try:
            # Input layers
            time_series = keras.layers.Input(shape=self.input_shape)
            category = keras.layers.Input(shape=(1,))

            # Encode categorical features
            category_embedding = keras.layers.Embedding(
                self.n_categories,
                self.hidden_dim // 4
            )(category)
            category_embedding = keras.layers.Flatten()(category_embedding)

            # Process time series with CNN layers
            x = keras.layers.Conv1D(
                filters=self.hidden_dim,
                kernel_size=3,
                padding='same',
                activation='relu'
            )(time_series)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.MaxPooling1D(pool_size=2)(x)

            # Add residual connections
            residual = x
            x = keras.layers.Conv1D(
                filters=self.hidden_dim,
                kernel_size=3,
                padding='same',
                activation='relu'
            )(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Add()([x, residual])

            # Global features
            x = keras.layers.GlobalAveragePooling1D()(x)

            # Combine with category embedding
            x = keras.layers.Concatenate()([x, category_embedding])

            # Dense layers with dropout
            x = keras.layers.Dense(self.hidden_dim, activation='relu')(x)
            x = keras.layers.Dropout(0.3)(x)
            x = keras.layers.Dense(self.hidden_dim // 2, activation='relu')(x)
            x = keras.layers.Dropout(0.2)(x)

            # Output layers
            reconstruction = keras.layers.Dense(
                self.input_shape[0] * self.input_shape[1],
                activation='linear'
            )(x)
            reconstruction = keras.layers.Reshape(self.input_shape)(reconstruction)

            # Create model
            self.model = keras.Model(
                inputs=[time_series, category],
                outputs=reconstruction
            )

            # Compile with improved optimizer configuration
            optimizer = keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                clipnorm=1.0  # Gradient clipping for stability
            )

            self.model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae']  # Add mean absolute error for monitoring
            )

            logger.info(f"Model architecture built successfully")
            self.model.summary(print_fn=logger.info)

        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise

    def train(self, timeseries_df: pd.DataFrame,
         relationships_df: pd.DataFrame,
         epochs: int = 10,
         threshold_percentile: float = 95,
         batch_size: int = 32) -> Dict[str, Any]:
        try:
            start_time = time.time()
            logger.info(f"Starting model training with {epochs} epochs")

            # Prepare data
            logger.info("Preparing training data...")

            # Select only numeric features for training
            numeric_data = timeseries_df[self.numeric_features].values
            numeric_data = numeric_data.reshape(-1, len(self.numeric_features)).astype(np.float32)

            # Calculate the number of complete sequences
            total_samples = len(numeric_data)
            sequence_length = self.input_shape[0]
            n_complete_sequences = total_samples // sequence_length
            samples_to_use = n_complete_sequences * sequence_length

            # Trim data to ensure complete sequences
            numeric_data = numeric_data[:samples_to_use]
            X = numeric_data.reshape(-1, *self.input_shape)

            # Process categories
            # Get unique services and their counts
            service_counts = relationships_df['service_type'].value_counts()
            unique_services = service_counts.index.tolist()

            # Fit the encoder on unique services
            self.category_encoder.fit(unique_services)

            # Create category labels
            service_indices = self.category_encoder.transform(relationships_df['service_type'].values)
            y = np.repeat(service_indices, samples_to_use // len(service_indices))

            # Ensure y matches X length
            if len(y) > len(X):
                y = y[:len(X)]
            elif len(y) < len(X):
                y = np.pad(y, (0, len(X) - len(y)), mode='wrap')

            # Convert to TensorFlow tensors
            X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
            y_tensor = tf.convert_to_tensor(y, dtype=tf.int32)

            # Build model if not already built
            if self.model is None:
                self._build_model()

            # Train with comprehensive monitoring
            history = self.model.fit(
                [X_tensor, y_tensor],
                X_tensor,  # Reconstruction target is the input
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )

            # Calculate reconstruction errors and threshold
            predictions = self.model.predict([X_tensor, y_tensor])
            reconstruction_errors = tf.reduce_mean(tf.square(X_tensor - predictions), axis=(1, 2))
            self.threshold = float(np.percentile(reconstruction_errors.numpy(), threshold_percentile))

            # Update training history
            training_time = time.time() - start_time
            self.training_history['loss'].extend(history.history['loss'])
            self.training_history['training_time'].append(training_time)

            self.is_trained = True

            logger.info(f"Training completed in {training_time:.2f} seconds")
            logger.info(f"Final loss: {history.history['loss'][-1]:.4f}")
            logger.info(f"Set reconstruction threshold: {self.threshold:.4f}")

            return {
                'history': history.history,
                'training_time': training_time,
                'final_loss': history.history['loss'][-1],
                'reconstruction_threshold': self.threshold
            }

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def predict(self, timeseries_df: pd.DataFrame,
               relationships_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before prediction")

            logger.info("Starting anomaly detection")
            prediction_start = time.time()

            # Prepare data with explicit type conversion
            numeric_data = timeseries_df[self.numeric_features].values
            numeric_data = numeric_data.reshape(-1, len(self.numeric_features)).astype(np.float32)

            # Calculate number of sequences
            total_samples = len(numeric_data)
            sequence_length = self.input_shape[0]
            n_complete_sequences = total_samples // sequence_length
            samples_to_use = n_complete_sequences * sequence_length

            # Trim data to ensure complete sequences
            numeric_data = numeric_data[:samples_to_use]
            X = numeric_data.reshape(-1, *self.input_shape)

            # Process categories
            service_indices = self.category_encoder.transform(relationships_df['service_type'].values)
            y = np.repeat(service_indices, samples_to_use // len(service_indices))

            # Ensure y matches X length
            if len(y) > len(X):
                y = y[:len(X)]
            elif len(y) < len(X):
                y = np.pad(y, (0, len(X) - len(y)), mode='wrap')

            # Convert to TensorFlow tensors
            X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
            y_tensor = tf.convert_to_tensor(y, dtype=tf.int32)

            # Get predictions
            predictions = self.model.predict([X_tensor, y_tensor])
            reconstruction_errors = tf.reduce_mean(tf.square(X_tensor - predictions), axis=(1, 2))
            reconstruction_errors = reconstruction_errors.numpy()

            # Detect anomalies
            anomalies = reconstruction_errors > self.threshold

            # Pattern analysis
            patterns = np.full_like(anomalies, fill_value='normal', dtype='object')
            for i in range(len(reconstruction_errors)):
                if anomalies[i]:
                    # Calculate volatility and trend
                    volatility = np.std(X[i])
                    trend = np.mean(np.diff(X[i]))
                    max_error_idx = np.argmax(np.mean(np.square(X[i] - predictions[i]), axis=1))

                    # Classify patterns
                    if volatility > np.mean(volatility):
                        if trend > 0:
                            patterns[i] = 'volatile_increasing'
                        else:
                            patterns[i] = 'volatile_decreasing'
                    else:
                        if max_error_idx < len(X[i]) // 2:
                            patterns[i] = 'early_deviation'
                        else:
                            patterns[i] = 'late_deviation'

            prediction_time = time.time() - prediction_start
            n_anomalies = int(np.sum(anomalies))

            logger.info(f"Detection completed in {prediction_time:.2f} seconds")
            logger.info(f"Found {n_anomalies} anomalies ({(n_anomalies/len(anomalies))*100:.2f}% of total)")

            if n_anomalies > 0:
                pattern_counts = np.unique(patterns[anomalies], return_counts=True)
                logger.info("Pattern distribution:")
                for pattern, count in zip(*pattern_counts):
                    logger.info(f"- {pattern}: {count}")

            return anomalies, patterns

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

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Save Keras model
            self.model.save(path)

            # Save additional properties
            properties = {
                'input_shape': self.input_shape,
                'n_categories': self.n_categories,
                'hidden_dim': self.hidden_dim,
                'learning_rate': self.learning_rate,
                'threshold': float(self.threshold),
                'is_trained': self.is_trained,
                'training_history': self.training_history,
                'category_encoder_classes': self.category_encoder.classes_.tolist()
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
            self.input_shape = tuple(properties['input_shape'])
            self.n_categories = properties['n_categories']
            self.hidden_dim = properties['hidden_dim']
            self.learning_rate = properties['learning_rate']
            self.threshold = properties['threshold']
            self.is_trained = properties['is_trained']
            self.training_history = properties['training_history']

            # Restore category encoder
            self.category_encoder.classes_ = np.array(properties['category_encoder_classes'])

            logger.info(f"Model and properties loaded from {path}")
            logger.info(f"Model training state: {'trained' if self.is_trained else 'untrained'}")

            return True

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise