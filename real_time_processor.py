import logging
import time
import os
from datetime import datetime, timedelta
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from collections import deque
from data_generator import CloudWatchDataGenerator
from time_gnn_model import TimeGNNAnomalyDetector
from visualization import TimeGNNVisualizer
import matplotlib.pyplot as plt

# Configure basic logging
logger = logging.getLogger(__name__)

class RealTimeProcessor:
    def __init__(self, window_size=100, update_interval=5, detection_threshold=1.5, model_type='timegnn'):
        """
        Initialize real-time processor for TimeGNN
        window_size: Number of data points to keep in memory
        update_interval: Seconds between updates
        detection_threshold: Threshold multiplier for anomaly detection
        model_type: Type of model to use ('timegnn' or 'structural')
        """
        self.window_size = window_size
        self.update_interval = update_interval
        self.detection_threshold = detection_threshold
        self.model_type = model_type
        self.data_generator = CloudWatchDataGenerator(n_timestamps=1)
        self.model = None
        self.current_threshold = None

        # Initialize sliding windows and statistics
        self.entities_df = None
        self.relationships_df = None
        self.timeseries_window = deque(maxlen=window_size)
        self.latest_anomalies = set()
        self.anomaly_history = deque(maxlen=window_size)

        # Statistics tracking
        self.total_anomalies = 0
        self.pattern_counts = {
            'spike': 0,
            'drop': 0,
            'gradual': 0,
            'pattern': 0
        }

        # Setup enhanced logging
        self._setup_logging()

        # Initialize visualizer
        self.visualizer = TimeGNNVisualizer()
        logger.info(f"Initialized RealTimeProcessor with:")
        logger.info(f"- Window Size: {window_size}")
        logger.info(f"- Update Interval: {update_interval}s")
        logger.info(f"- Detection Threshold: {detection_threshold}")
        logger.info(f"- Model Type: {model_type}")

    def _setup_logging(self):
        """Configure detailed logging for real-time monitoring"""
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Add file handler for persistent logging
        os.makedirs('logs', exist_ok=True)
        file_handler = logging.FileHandler('logs/realtime_processor.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def _log_statistics(self):
        """Log current detection statistics"""
        logger.info("\n" + "="*50)
        logger.info("REAL-TIME MONITORING STATISTICS")
        logger.info("="*50)
        logger.info(f"Total Anomalies Detected: {self.total_anomalies}")
        logger.info("\nPattern Distribution:")
        for pattern, count in self.pattern_counts.items():
            if count > 0:
                percentage = (count / self.total_anomalies * 100) if self.total_anomalies > 0 else 0
                logger.info(f"- {pattern.upper()}: {count} ({percentage:.1f}%)")

        current_anomalies = len(self.latest_anomalies)
        if current_anomalies > 0:
            logger.info(f"\nCurrent Anomalous Entities: {sorted(self.latest_anomalies)}")
        logger.info("="*50 + "\n")

    def update_visualizations(self):
        """Update real-time visualizations for anomalies using matplotlib"""
        if not self.latest_anomalies and not self.anomaly_history:
            return

        try:
            # Create anomaly pattern dictionary from latest detection
            anomaly_patterns = {}
            if self.anomaly_history:
                latest_history = list(self.anomaly_history)[-1]
                if 'anomalies' in latest_history:
                    anomaly_patterns.update(latest_history['anomalies'])

            # Update time series plots for anomalous entities
            current_data = pd.DataFrame(list(self.timeseries_window))

            if not current_data.empty and self.latest_anomalies:
                plt.figure(figsize=(12, 6))
                for entity_id in self.latest_anomalies:
                    entity_data = current_data[current_data['entity_id'] == entity_id]
                    if not entity_data.empty:
                        plt.plot(entity_data['timestamp'], entity_data['CPUUtilization'], 
                               label=f'Entity {entity_id}')

                plt.title('Anomalous Entities Time Series')
                plt.xlabel('Timestamp')
                plt.ylabel('CPU Utilization')
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()

                # Save the plot
                os.makedirs('visualizations', exist_ok=True)
                plt.savefig(os.path.join('visualizations', 'anomaly_timeseries.png'))
                plt.close()

            logger.info(f"Updated visualizations for {len(self.latest_anomalies)} anomalous entities")

        except Exception as e:
            logger.error(f"Error updating visualizations: {str(e)}")
            logger.error("Continuing processing despite visualization error")

    def process_new_data(self):
        """Process a new batch of data"""
        try:
            # Generate new data point
            new_data = self.data_generator.generate_time_series()
            self.timeseries_window.extend(new_data.to_dict('records'))

            # Convert window to DataFrame
            current_data = pd.DataFrame(list(self.timeseries_window))

            # Detect anomalies using the model
            predictions, pattern_types = self.model.predict(
                current_data,
                self.relationships_df
            )

            # Update anomaly tracking
            anomaly_entities = []
            for i, (is_anomaly, pattern) in enumerate(zip(predictions, pattern_types)):
                if is_anomaly:
                    entity_id = current_data['entity_id'].unique()[i]
                    anomaly_entities.append((entity_id, pattern))
                    self.pattern_counts[pattern] += 1
                    self.total_anomalies += 1

            self.latest_anomalies = {entity for entity, _ in anomaly_entities}

            # Store anomaly history
            self.anomaly_history.append({
                'timestamp': datetime.now(),
                'anomalies': {
                    entity: pattern for entity, pattern in anomaly_entities
                }
            })

            # Log statistics
            self._log_statistics()

            return len(self.latest_anomalies)

        except Exception as e:
            logger.error(f"Error processing new data: {str(e)}")
            return 0

    def initialize_model(self):
        """Initialize and train the TimeGNN model with initial data"""
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                logger.info("Generating initial data for model training...")

                # Generate initial entity metadata and relationships
                self.entities_df = self.data_generator.generate_entity_metadata()
                self.relationships_df = self.data_generator.generate_relationships()

                # Generate initial time series data
                initial_data = self.data_generator.generate_time_series()
                self.timeseries_window.extend(initial_data.to_dict('records'))

                logger.info(f"Generated initial data with {len(self.entities_df)} entities")

                # Initialize TimeGNN model
                n_features = 4  # CPU, Memory, NetworkIn, NetworkOut
                n_categories = len(self.entities_df['service_type'].unique())
                self.model = TimeGNNAnomalyDetector(n_features, n_categories)

                # Prepare initial training data
                training_data = pd.DataFrame(list(self.timeseries_window))

                # Train model with threshold percentile based on detection_threshold
                logger.info("Training initial TimeGNN model...")
                # Convert detection_threshold to percentile (1.5 -> 93.3, 2.0 -> 95.4, etc.)
                threshold_percentile = int(90 - (100 / (self.detection_threshold * 2)))  # More sensitive threshold
                history = self.model.train(
                    training_data,
                    self.relationships_df,
                    epochs=5,
                    threshold_percentile=threshold_percentile
                )

                logger.info(f"Model initialized with threshold_percentile: {threshold_percentile}")
                logger.info(f"Training completed. Final loss: {history.history['loss'][-1]:.3f}")
                return True

            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    logger.error("Maximum retries reached. Model initialization failed.")
                    raise

    def run(self):
        """Main real-time processing loop"""
        logger.info(f"Starting real-time processing with model type: {self.model_type}")

        try:
            # Initialize model with initial data
            if not self.initialize_model():
                logger.error("Failed to initialize model")
                return

            iteration = 0
            logger.info("\nStarting monitoring loop...")
            logger.info("Press Ctrl+C to stop monitoring\n")

            while True:
                start_time = time.time()
                iteration += 1

                try:
                    # Process new data
                    n_anomalies = self.process_new_data()

                    # Update visualizations if anomalies detected
                    if n_anomalies > 0:
                        self.update_visualizations()

                    # Maintain update interval
                    elapsed = time.time() - start_time
                    sleep_time = max(0, self.update_interval - elapsed)

                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    else:
                        logger.warning(f"Processing took longer than update interval by {-sleep_time:.2f} seconds")

                except Exception as e:
                    logger.error(f"Error in iteration {iteration}: {str(e)}")
                    logger.error("Continuing to next iteration...")
                    time.sleep(self.update_interval)

        except KeyboardInterrupt:
            logger.info("\nReceived keyboard interrupt, stopping real-time processing...")
            self._log_statistics()  # Final statistics
        except Exception as e:
            logger.error(f"Fatal error in real-time processing: {str(e)}")
            raise