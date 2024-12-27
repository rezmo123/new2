import os
import argparse
import logging
import sys
from datetime import datetime
import numpy as np
import pandas as pd
from time_gnn_model import TimeGNNAnomalyDetector
from data_generator import CloudWatchDataGenerator
from typing import Optional, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

class AnomalyDetectionCLI:
    def __init__(self):
        """Initialize the TimeGNN-based anomaly detection system"""
        self.data_generator = CloudWatchDataGenerator(n_entities=10, n_timestamps=100)
        self.time_gnn_model: Optional[TimeGNNAnomalyDetector] = None
        self.entities_df: Optional[pd.DataFrame] = None
        self.relationships_df: Optional[pd.DataFrame] = None
        self.timeseries_df: Optional[pd.DataFrame] = None
        logger.info("Initialized TimeGNN Anomaly Detection System")

    def generate_data(self) -> bool:
        """Generate synthetic data for TimeGNN training"""
        try:
            logger.info("Generating synthetic data...")
            self.entities_df = self.data_generator.generate_entity_metadata()
            self.relationships_df = self.data_generator.generate_relationships()
            self.timeseries_df = self.data_generator.generate_time_series()

            # Log data generation statistics
            logger.info(f"Generated {len(self.entities_df)} entities")
            logger.info(f"Generated {len(self.relationships_df)} relationships")
            logger.info(f"Generated {len(self.timeseries_df)} time series records")

            return True
        except Exception as e:
            logger.error(f"Error generating data: {str(e)}", exc_info=True)
            return False

    def train_model(self, threshold_percentile: float = 95) -> Optional[Any]:
        """Train the TimeGNN model"""
        if not all([self.entities_df is not None, 
                   self.relationships_df is not None, 
                   self.timeseries_df is not None]):
            logger.error("No data available for training. Generate data first!")
            return None

        try:
            logger.info(f"Training TimeGNN model with threshold percentile: {threshold_percentile}")
            n_features = 4  # CPU, Memory, NetworkIn, NetworkOut
            n_categories = len(self.entities_df['service_type'].unique())

            # Initialize TimeGNN model
            self.time_gnn_model = TimeGNNAnomalyDetector(
                input_shape=(5, n_features),  # sequence_length=5, n_features=4
                n_categories=n_categories
            )

            # Train model
            history = self.time_gnn_model.train(
                timeseries_df=self.timeseries_df,
                relationships_df=self.relationships_df,
                epochs=10,
                threshold_percentile=threshold_percentile
            )

            logger.info("Model training completed successfully")
            return history

        except Exception as e:
            logger.error(f"Error training model: {str(e)}", exc_info=True)
            return None

    def detect_anomalies(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect anomalies using the trained TimeGNN model"""
        try:
            if self.time_gnn_model is None:
                logger.error("Model not initialized. Train the model first!")
                return None, None

            if not all([self.timeseries_df is not None, self.relationships_df is not None]):
                logger.error("Required data is missing")
                return None, None

            logger.info("Starting anomaly detection...")
            predictions, patterns = self.time_gnn_model.predict(
                self.timeseries_df,
                self.relationships_df
            )

            # Get unique entities
            unique_entities = self.timeseries_df['entity_id'].unique()
            n_entities = len(unique_entities)

            # Ensure predictions and patterns match entity count
            predictions = np.array(predictions)[:n_entities]
            patterns = np.array(patterns)[:n_entities]

            # Log detection results
            n_anomalies = int(np.sum(predictions))
            logger.info(f"\nDetection Summary:")
            logger.info(f"Total entities analyzed: {len(predictions)}")
            logger.info(f"Anomalies detected: {n_anomalies}")

            if n_anomalies > 0:
                anomaly_results = pd.DataFrame({
                    'entity_id': unique_entities,
                    'is_anomaly': predictions,
                    'pattern_type': patterns
                })

                pattern_counts = anomaly_results[anomaly_results['is_anomaly']]['pattern_type'].value_counts()
                logger.info("\nAnomaly Patterns:")
                for pattern, count in pattern_counts.items():
                    logger.info(f"{pattern}: {count} instances")

            return predictions, patterns

        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return None, None

def main():
    parser = argparse.ArgumentParser(description='TimeGNN Anomaly Detection System')
    parser.add_argument('--action', choices=['train', 'detect'], required=True,
                      help='Action to perform: train (train new model) or detect (detect anomalies)')
    parser.add_argument('--threshold', type=float, default=95.0,
                      help='Threshold percentile for anomaly detection (default: 95.0)')

    args = parser.parse_args()

    try:
        # Initialize system
        detector = AnomalyDetectionCLI()

        # Generate data (required for both training and detection)
        if not detector.generate_data():
            logger.error("Failed to generate data")
            sys.exit(1)

        if args.action == 'train':
            # Train the model
            if detector.train_model(args.threshold) is not None:
                logger.info("Training completed successfully")
            else:
                logger.error("Training failed")
                sys.exit(1)

        elif args.action == 'detect':
            # Detect anomalies
            predictions, patterns = detector.detect_anomalies()
            if predictions is not None:
                logger.info("Anomaly detection completed successfully")
            else:
                logger.error("Anomaly detection failed")
                sys.exit(1)

    except Exception as e:
        logger.error(f"Error during execution: {str(e)}", exc_info=True)
        sys.exit(1)

    logger.info("Process completed successfully")

if __name__ == "__main__":
    main()