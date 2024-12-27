import logging
import argparse
import time
from typing import Optional, Dict, Any
import sys
import os
import pandas as pd

# Configure path to include core directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from core.time_gnn_model import TimeGNNAnomalyDetector
from core.models.structural_gnn_model import StructuralGNNDetector
from core.data_generator import CloudWatchDataGenerator

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('anomaly_detection_demo.log')
    ]
)
logger = logging.getLogger(__name__)

def run_timegnn_demo(threshold_percentile: float = 95.0, epochs: int = 10) -> Dict[str, Any]:
    """Run TimeGNN demonstration with comprehensive error handling and logging"""
    try:
        start_time = time.time()
        logger.info("Starting TimeGNN demonstration...")

        # Load test data with detailed validation
        try:
            entities_df = pd.read_csv('test_entities.csv')
            relationships_df = pd.read_csv('test_relationships.csv')
            timeseries_df = pd.read_csv('test_timeseries.csv')

            # Convert timestamp column to datetime
            timeseries_df['timestamp'] = pd.to_datetime(timeseries_df['timestamp'])

            # Log data shapes and sample rows
            logger.info(f"Loaded {len(entities_df)} entities")
            logger.info(f"Entity columns: {entities_df.columns.tolist()}")
            logger.info(f"Sample entities:\n{entities_df.head()}")

            logger.info(f"Loaded {len(relationships_df)} relationships")
            logger.info(f"Relationship columns: {relationships_df.columns.tolist()}")
            logger.info(f"Sample relationships:\n{relationships_df.head()}")

            logger.info(f"Loaded {len(timeseries_df)} time series records")
            logger.info(f"Time series columns: {timeseries_df.columns.tolist()}")
            logger.info(f"Sample time series:\n{timeseries_df.head()}")

            logger.info("Successfully loaded test data")
        except Exception as e:
            logger.error(f"Failed to load test data: {str(e)}")
            raise

        # Initialize and train model with enhanced logging
        try:
            n_features = len(['CPUUtilization', 'MemoryUtilization', 'NetworkIn', 'NetworkOut'])
            n_categories = len(entities_df['service_type'].unique())
            logger.info(f"Initializing TimeGNN with {n_features} features and {n_categories} service types")

            model = TimeGNNAnomalyDetector(
                input_shape=(5, n_features),
                n_categories=n_categories
            )
            logger.info("Successfully initialized TimeGNN model")
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise

        # Train model with comprehensive monitoring
        try:
            history = model.train(
                timeseries_df=timeseries_df,
                relationships_df=relationships_df,
                epochs=epochs,
                threshold_percentile=threshold_percentile
            )
            logger.info("Successfully trained TimeGNN model")

            if isinstance(history, dict) and 'history' in history:
                final_loss = history['history'].get('loss', [])[-1] if history['history'].get('loss') else None
                logger.info(f"Training metrics - Final loss: {final_loss:.4f}")
        except Exception as e:
            logger.error(f"Failed to train model: {str(e)}")
            raise

        # Detect anomalies with enhanced error handling
        try:
            predictions, patterns = model.predict(timeseries_df, relationships_df)
            n_anomalies = int(predictions.sum()) if predictions is not None else 0
            unique_patterns = list(set(patterns[predictions])) if predictions is not None and patterns is not None else []

            logger.info(f"Detected {n_anomalies} anomalies ({(n_anomalies/len(predictions))*100:.2f}% of total)")
            logger.info(f"Pattern types found: {unique_patterns}")
            logger.info("Successfully completed anomaly detection")
        except Exception as e:
            logger.error(f"Failed to detect anomalies: {str(e)}")
            raise

        runtime = time.time() - start_time

        results = {
            'model_type': 'TimeGNN',
            'runtime': runtime,
            'n_anomalies': n_anomalies,
            'patterns': unique_patterns,
            'training_history': history.get('history', {}) if isinstance(history, dict) else history
        }

        logger.info(f"TimeGNN demo completed in {runtime:.2f} seconds")
        return results

    except Exception as e:
        logger.error(f"TimeGNN demo failed: {str(e)}")
        return {
            'model_type': 'TimeGNN',
            'error': str(e),
            'runtime': time.time() - start_time
        }

def run_structuralgnn_demo(threshold_percentile: float = 95.0, epochs: int = 10) -> Dict[str, Any]:
    """Run StructuralGNN demonstration with comprehensive error handling and logging"""
    try:
        logger.info("Starting StructuralGNN demonstration...")
        start_time = time.time()

        # Load test data
        try:
            relationships_df = pd.read_csv('test_relationships.csv')
            # Log data details
            logger.info(f"Loaded {len(relationships_df)} relationships")
            logger.info(f"Relationship DataFrame columns: {relationships_df.columns.tolist()}")
            logger.info(f"First few relationships:\n{relationships_df.head()}")
            logger.info("Successfully loaded structural data")
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise

        # Initialize model with error handling
        try:
            model = StructuralGNNDetector()
            logger.info("Successfully initialized StructuralGNN model")
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise

        # Train model with error handling
        try:
            history = model.train(
                relationships_df=relationships_df,
                epochs=epochs,
                threshold_percentile=threshold_percentile
            )
            logger.info("Successfully trained StructuralGNN model")
        except Exception as e:
            logger.error(f"Failed to train model: {str(e)}")
            raise

        # Detect anomalies with error handling
        try:
            anomalies, patterns, scores = model.predict(relationships_df)
            logger.info("Successfully completed anomaly detection")
        except Exception as e:
            logger.error(f"Failed to detect anomalies: {str(e)}")
            raise

        runtime = time.time() - start_time

        # Handle training history with the same care as TimeGNN
        if history is None:
            training_history = None
        elif isinstance(history, dict):
            training_history = history.get('history', history)
        else:
            training_history = history.history if hasattr(history, 'history') else None

        results = {
            'model_type': 'StructuralGNN',
            'runtime': runtime,
            'n_anomalies': int(anomalies.sum()) if anomalies is not None else 0,
            'patterns': list(set(patterns[anomalies])) if anomalies is not None and patterns is not None else [],
            'training_history': training_history,
            'reconstruction_scores': scores.tolist() if scores is not None else None
        }

        logger.info(f"StructuralGNN demo completed in {runtime:.2f} seconds")
        logger.info(f"Detected {results['n_anomalies']} anomalies")
        logger.info(f"Pattern types found: {results['patterns']}")

        return results

    except Exception as e:
        logger.error(f"StructuralGNN demo failed: {str(e)}")
        return {
            'model_type': 'StructuralGNN',
            'error': str(e),
            'runtime': time.time() - start_time
        }

def compare_models(results: Dict[str, Dict[str, Any]]) -> None:
    """Compare performance metrics between models with detailed logging"""
    logger.info("\nModel Comparison Summary:")
    logger.info("-" * 50)

    metrics_log = []
    for model_type, metrics in results.items():
        if 'error' not in metrics:
            metrics_log.append(f"\n{model_type}:")
            metrics_log.append(f"Runtime: {metrics['runtime']:.2f} seconds")
            metrics_log.append(f"Anomalies detected: {metrics['n_anomalies']}")
            metrics_log.append(f"Pattern types: {metrics['patterns']}")

            if metrics.get('training_history'):
                final_loss = next(
                    (v[-1] for k, v in metrics['training_history'].items() if k == 'loss'),
                    None
                )
                if final_loss is not None:
                    metrics_log.append(f"Final training loss: {final_loss:.4f}")
        else:
            metrics_log.append(f"\n{model_type}: Failed - {metrics['error']}")

    logger.info("\n".join(metrics_log))
    logger.info("-" * 50)

def run_demo(model_type: str = 'all', threshold_percentile: float = 95.0, epochs: int = 10) -> bool:
    """Run demonstration of the specified anomaly detection model(s)"""
    try:
        results = {}

        if model_type in ['time', 'all']:
            results['TimeGNN'] = run_timegnn_demo(threshold_percentile, epochs)

        if model_type in ['structural', 'all']:
            results['StructuralGNN'] = run_structuralgnn_demo(threshold_percentile, epochs)

        if len(results) > 1:
            compare_models(results)

        # Verify all models completed successfully
        success = all('error' not in result for result in results.values())
        if success:
            logger.info("Demo completed successfully!")
        else:
            logger.warning("Demo completed with some failures")

        return success

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Production-ready Anomaly Detection Demo')
    parser.add_argument('--model', type=str, default='all',
                      choices=['time', 'structural', 'all'],
                      help='Model to demonstrate (default: all)')
    parser.add_argument('--threshold', type=float, default=95.0,
                      help='Threshold percentile for anomaly detection (default: 95.0)')
    parser.add_argument('--epochs', type=int, default=10,
                      help='Number of training epochs (default: 10)')

    args = parser.parse_args()
    success = run_demo(args.model, args.threshold, args.epochs)
    sys.exit(0 if success else 1)