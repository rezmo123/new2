import logging
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from concurrent.futures import ThreadPoolExecutor, as_completed
from time_gnn_model import TimeGNNAnomalyDetector
from structural_gnn_model import StructuralGNNDetector
from visualization import TimeGNNVisualizer
import matplotlib.pyplot as plt  # type: ignore
from typing import Any, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridAnomalyDetector:
    def __init__(self):
        """Initialize hybrid detector with both temporal and structural models"""
        self.time_gnn = None
        self.structural_gnn = None
        self.visualizer = TimeGNNVisualizer()
        self.initialized = False
        self.thread_pool = ThreadPoolExecutor(max_workers=2)  # One for each model

    def initialize_models(self, n_temporal_features, n_categories):
        """Initialize both GNN models in parallel"""
        try:
            # Initialize models in parallel
            future_time_gnn = self.thread_pool.submit(
                lambda: TimeGNNAnomalyDetector(n_temporal_features, n_categories)
            )
            future_structural = self.thread_pool.submit(
                lambda: StructuralGNNDetector()
            )

            # Wait for both initializations to complete
            self.time_gnn = future_time_gnn.result()
            self.structural_gnn = future_structural.result()
            self.initialized = True
            logger.info("Hybrid detector initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    def visualize_detection_results(self, results: pd.DataFrame, entities_df: pd.DataFrame, 
                                  relationships_df: pd.DataFrame, timeseries_df: pd.DataFrame) -> None:
        """Visualize anomaly detection results using both entity graphs and time series plots"""
        try:
            # Create combined anomaly dictionary
            anomaly_dict = {}
            anomalous_entities = results[
                results['temporal_anomaly'] | results['structural_anomaly']
            ]

            for _, row in anomalous_entities.iterrows():
                entity_id = str(row['entity_id'])
                patterns = []
                if row['temporal_anomaly']:
                    patterns.append(f"T:{row['temporal_pattern']}")
                if row['structural_anomaly']:
                    patterns.append(f"S:{row['structural_pattern']}")
                anomaly_dict[entity_id] = ' & '.join(patterns)

            # Plot entity graph with anomalies
            graph_fig = self.visualizer.plot_entity_graph(
                entities_df,
                relationships_df,
                anomaly_dict
            )

            if graph_fig is not None:
                logger.info("Successfully generated entity graph visualization")
                # Save or display graph as needed
                plt.close(graph_fig)

            # Plot time series for anomalous entities (limit to top 3 by reconstruction error)
            top_anomalies = anomalous_entities.nlargest(3, 'reconstruction_error')

            for _, row in top_anomalies.iterrows():
                entity_id = str(row['entity_id'])
                ts_fig = self.visualizer.plot_time_series(
                    timeseries_df,
                    entity_id
                )
                if ts_fig is not None:
                    logger.info(f"Successfully generated time series plot for entity {entity_id}")
                    # Save or display time series as needed
                    plt.close(ts_fig)

        except Exception as e:
            logger.error(f"Error visualizing detection results: {str(e)}")

    def detect_anomalies(self, timeseries_df: pd.DataFrame, relationships_df: pd.DataFrame,
                        entities_df: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
        """Detect both temporal and structural anomalies in parallel"""
        if not self.initialized:
            raise ValueError("Models not initialized. Call initialize_models first.")

        try:
            # Submit detection tasks to thread pool
            logger.info("\nStarting parallel anomaly detection...")
            future_temporal = self.thread_pool.submit(
                self.time_gnn.predict,
                timeseries_df,
                relationships_df
            )
            future_structural = self.thread_pool.submit(
                self.structural_gnn.predict,
                relationships_df
            )

            # Wait for both detection tasks to complete
            temporal_predictions, temporal_patterns = future_temporal.result()
            structural_predictions, structural_patterns, reconstruction_errors = future_structural.result()

            # Validate array lengths before creating DataFrame
            entity_ids = timeseries_df['entity_id'].unique()
            n_entities = len(entity_ids)

            # Convert all inputs to numpy arrays
            temporal_predictions = np.asarray(temporal_predictions) if temporal_predictions is not None else np.array([])
            temporal_patterns = np.asarray(temporal_patterns) if temporal_patterns is not None else np.array([])
            structural_predictions = np.asarray(structural_predictions) if structural_predictions is not None else np.array([])
            structural_patterns = np.asarray(structural_patterns) if structural_patterns is not None else np.array([])
            reconstruction_errors = np.asarray(reconstruction_errors) if reconstruction_errors is not None else np.array([])

            # Ensure all arrays have the same length
            def pad_or_truncate(arr: np.ndarray, target_length: int, fill_value: Any = False) -> np.ndarray:
                if len(arr) > target_length:
                    return arr[:target_length]
                elif len(arr) < target_length:
                    pad_width = target_length - len(arr)
                    if isinstance(fill_value, str):
                        padded = np.array([fill_value] * target_length)
                        padded[:len(arr)] = arr
                        return padded
                    return np.pad(arr, (0, pad_width), constant_values=fill_value)
                return arr

            temporal_predictions = pad_or_truncate(temporal_predictions, n_entities)
            temporal_patterns = pad_or_truncate(temporal_patterns, n_entities, 'normal')
            structural_predictions = pad_or_truncate(structural_predictions, n_entities)
            structural_patterns = pad_or_truncate(structural_patterns, n_entities, 'normal')
            reconstruction_errors = pad_or_truncate(reconstruction_errors, n_entities, 0.0)

            # Create results DataFrame
            try:
                results = pd.DataFrame({
                    'entity_id': entity_ids,
                    'temporal_anomaly': temporal_predictions,
                    'temporal_pattern': temporal_patterns,
                    'structural_anomaly': structural_predictions,
                    'structural_pattern': structural_patterns,
                    'reconstruction_error': reconstruction_errors
                })

                # Generate visualizations if entities_df is provided
                if entities_df is not None:
                    self.visualize_detection_results(
                        results,
                        entities_df,
                        relationships_df,
                        timeseries_df
                    )

                return results

            except Exception as e:
                logger.error(f"Error creating results DataFrame: {str(e)}")
                return None

        except Exception as e:
            logger.error(f"Error during hybrid detection: {str(e)}")
            return None

    def train(self, timeseries_df: pd.DataFrame, relationships_df: pd.DataFrame,
              epochs: int = 10) -> Tuple[Any, Any]:
        """Train both models in parallel"""
        if not self.initialized:
            raise ValueError("Models not initialized. Call initialize_models first.")

        try:
            # Submit training tasks to thread pool
            logger.info("Training both models in parallel...")
            future_time = self.thread_pool.submit(
                self.time_gnn.train,
                timeseries_df,
                relationships_df,
                epochs=epochs
            )
            future_structural = self.thread_pool.submit(
                self.structural_gnn.train,
                relationships_df,
                epochs=epochs
            )

            # Wait for both training tasks to complete
            time_history = future_time.result()
            structural_history = future_structural.result()

            return time_history, structural_history

        except Exception as e:
            logger.error(f"Error during hybrid training: {str(e)}")
            raise

    def _pad_or_truncate(self, array, target_length, fill_value=False):
        """Helper function to ensure arrays have consistent length"""
        try:
            # Convert to numpy array if not already
            if not isinstance(array, np.ndarray):
                array = np.array(array)

            # Handle empty array
            if array.size == 0:
                if isinstance(fill_value, str):
                    return np.array([fill_value] * target_length)
                return np.full(target_length, fill_value)

            current_length = len(array)

            # Truncate if longer
            if current_length > target_length:
                return array[:target_length]

            # Pad if shorter
            elif current_length < target_length:
                if isinstance(fill_value, str):
                    # For string arrays (patterns)
                    padded = np.array([fill_value] * target_length)
                    padded[:current_length] = array
                    return padded
                else:
                    # For numeric arrays
                    return np.pad(
                        array,
                        (0, target_length - current_length),
                        mode='constant',
                        constant_values=fill_value
                    )

            return array
        except Exception as e:
            logger.error(f"Error in _pad_or_truncate: {str(e)}")
            if isinstance(fill_value, str):
                return np.array([fill_value] * target_length)
            return np.full(target_length, fill_value)

    def _print_detection_report(self, results, timeseries_df):
        """Print comprehensive anomaly detection report"""
        logger.info("\n" + "="*80)
        logger.info("HYBRID ANOMALY DETECTION REPORT")
        logger.info("="*80)

        # Overall statistics
        total_entities = len(results)
        temporal_anomalies = results['temporal_anomaly'].sum()
        structural_anomalies = results['structural_anomaly'].sum()
        combined_anomalies = results[
            results['temporal_anomaly'] | results['structural_anomaly']
        ].shape[0]

        logger.info("\n📊 DETECTION STATISTICS")
        logger.info("-"*60)
        logger.info(f"Total Entities Monitored: {total_entities}")
        logger.info(f"Temporal Anomalies: {temporal_anomalies}")
        logger.info(f"Structural Anomalies: {structural_anomalies}")
        logger.info(f"Combined Anomalies: {combined_anomalies}")

        if combined_anomalies > 0:
            # Pattern analysis
            logger.info("\n🔍 ANOMALY PATTERN DISTRIBUTION")
            logger.info("-"*60)

            # Temporal patterns
            temporal_patterns = results[
                results['temporal_anomaly']
            ]['temporal_pattern'].value_counts()

            logger.info("\n⏰ Temporal Patterns:")
            for pattern, count in temporal_patterns.items():
                percentage = (count / temporal_anomalies) * 100
                logger.info(f"  • {pattern.upper()}: {count} ({percentage:.1f}%)")

            # Structural patterns
            structural_patterns = results[
                results['structural_anomaly']
            ]['structural_pattern'].value_counts()

            logger.info("\n🔗 Structural Patterns:")
            for pattern, count in structural_patterns.items():
                percentage = (count / structural_anomalies) * 100
                logger.info(f"  • {pattern.upper()}: {count} ({percentage:.1f}%)")

            # Correlation analysis
            both_anomalous = results[
                results['temporal_anomaly'] & results['structural_anomaly']
            ].shape[0]
            if both_anomalous > 0:
                logger.info("\n🔄 Pattern Correlation")
                logger.info("-"*60)
                correlation_rate = (both_anomalous / combined_anomalies) * 100
                logger.info(f"Entities with both anomaly types: {both_anomalous} ({correlation_rate:.1f}%)")

            # Detailed entity analysis
            logger.info("\n📋 DETAILED ENTITY ANALYSIS")
            logger.info("="*80)

            anomalous_entities = results[
                results['temporal_anomaly'] | results['structural_anomaly']
            ]

            for _, row in anomalous_entities.iterrows():
                logger.info(f"\n📌 Entity: {row['entity_id']}")

                if row['temporal_anomaly']:
                    logger.info("  ⏰ Temporal Anomaly:")
                    logger.info(f"    Type: {row['temporal_pattern']}")

                    # Get latest metrics
                    entity_metrics = timeseries_df[
                        timeseries_df['entity_id'] == row['entity_id']
                    ].iloc[-1]

                    logger.info("    Current Metrics:")
                    logger.info(f"      • CPU: {entity_metrics['CPUUtilization']:.1f}%")
                    logger.info(f"      • Memory: {entity_metrics['MemoryUtilization']:.1f}%")
                    logger.info(f"      • Network In: {entity_metrics['NetworkIn']:.1f}")
                    logger.info(f"      • Network Out: {entity_metrics['NetworkOut']:.1f}")

                if row['structural_anomaly']:
                    logger.info("  🔗 Structural Anomaly:")
                    logger.info(f"    Type: {row['structural_pattern']}")

        logger.info("\n" + "="*80)
        logger.info("END OF HYBRID ANOMALY DETECTION REPORT")
        logger.info("="*80 + "\n")