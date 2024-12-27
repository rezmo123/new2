import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class TimeSeriesPreprocessor:
    def __init__(self, sequence_length=5):
        """Initialize TimeSeriesPreprocessor with specified sequence length"""
        self.sequence_length = sequence_length
        self.feature_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = ['CPUUtilization', 'MemoryUtilization', 'NetworkIn', 'NetworkOut']
        logger.info(f"Initialized TimeSeriesPreprocessor with sequence_length={sequence_length}")

    def _convert_to_numeric(self, df):
        """Convert string columns to numeric values with improved error handling"""
        try:
            numeric_df = df.copy()
            for feature in self.feature_names:
                if feature in df.columns:
                    if numeric_df[feature].dtype == 'object':
                        numeric_df[feature] = numeric_df[feature].str.replace(
                            r'[^\d.-]+', '', regex=True
                        ).str.strip()

                    numeric_df[feature] = pd.to_numeric(numeric_df[feature], errors='coerce')
                    nan_count = numeric_df[feature].isna().sum()
                    if nan_count > 0:
                        logger.warning(f"{nan_count} values in {feature} could not be converted to numeric")
                        mean_value = numeric_df[feature].mean()
                        numeric_df[feature] = numeric_df[feature].fillna(mean_value)

                    numeric_df[feature] = numeric_df[feature].astype(np.float32)

                    if feature in ['CPUUtilization', 'MemoryUtilization']:
                        out_of_range = numeric_df[feature] > 100
                        if out_of_range.any():
                            logger.warning(f"Found {out_of_range.sum()} values > 100% in {feature}")
                            numeric_df.loc[out_of_range, feature] = 100.0

            return numeric_df

        except Exception as e:
            logger.error(f"Error converting to numeric values: {str(e)}")
            raise

    def normalize_sequences(self, sequences, target_length=None):
        """Normalize sequence lengths through padding or truncation"""
        try:
            if sequences is None or len(sequences) == 0:
                logger.warning("Empty sequences provided to normalize_sequences")
                return np.array([])

            if target_length is None:
                target_length = self.sequence_length

            # Input validation
            if not isinstance(sequences, np.ndarray):
                sequences = np.array(sequences)

            if len(sequences.shape) != 3:
                raise ValueError(f"Expected 3D array (n_sequences, sequence_length, n_features), got shape {sequences.shape}")

            n_sequences, curr_length, n_features = sequences.shape
            logger.info(f"Normalizing {n_sequences} sequences from length {curr_length} to {target_length}")

            normalized_sequences = np.zeros((n_sequences, target_length, n_features), dtype=np.float32)

            for i, seq in enumerate(sequences):
                try:
                    # Validate sequence data
                    if not np.all(np.isfinite(seq)):
                        logger.warning(f"Found non-finite values in sequence {i}, replacing with zeros")
                        seq = np.nan_to_num(seq, 0)

                    seq_length = min(curr_length, target_length)

                    # Copy valid data
                    normalized_sequences[i, :seq_length] = seq[:seq_length]

                    # Zero-pad if needed
                    if seq_length < target_length:
                        normalized_sequences[i, seq_length:] = np.zeros((target_length - seq_length, n_features))

                except Exception as e:
                    logger.error(f"Error processing sequence {i}: {str(e)}")
                    # Initialize with zeros for failed sequences
                    normalized_sequences[i] = np.zeros((target_length, n_features))

            # Final validation
            if not np.all(np.isfinite(normalized_sequences)):
                logger.warning("Found non-finite values after normalization, replacing with zeros")
                normalized_sequences = np.nan_to_num(normalized_sequences, 0)

            logger.info(f"Successfully normalized sequences to shape {normalized_sequences.shape}")
            return normalized_sequences

        except Exception as e:
            logger.error(f"Error in normalize_sequences: {str(e)}")
            raise

    def prepare_temporal_data(self, timeseries_df, entities_df):
        """Prepare temporal sequences with improved anomaly detection sensitivity"""
        try:
            logger.info("Starting temporal data preparation...")

            if timeseries_df is None or entities_df is None:
                raise ValueError("Input DataFrames cannot be empty")

            # Convert timestamps to seconds since epoch
            if 'timestamp' in timeseries_df.columns:
                if isinstance(timeseries_df['timestamp'].iloc[0], str):
                    timeseries_df['timestamp'] = pd.to_datetime(timeseries_df['timestamp'])
                if isinstance(timeseries_df['timestamp'].iloc[0], pd.Timestamp):
                    timeseries_df['timestamp'] = timeseries_df['timestamp'].astype(np.int64) // 10**9

            # Convert features to numeric with mean imputation
            numeric_df = self._convert_to_numeric(timeseries_df)

            # Sort data by entity and timestamp
            numeric_df = numeric_df.sort_values(['entity_id', 'timestamp'])

            # Extract features and scale
            feature_data = numeric_df[self.feature_names].values.astype(np.float32)
            scaled_features = self.feature_scaler.fit_transform(feature_data)

            # Create sequences
            sequences = []
            categories = []

            for entity_id in numeric_df['entity_id'].unique():
                entity_mask = numeric_df['entity_id'] == entity_id
                entity_data = scaled_features[entity_mask]

                if len(entity_data) < self.sequence_length:
                    continue

                # Create sequences with proper float32 dtype
                for i in range(len(entity_data) - self.sequence_length + 1):
                    seq = entity_data[i:i + self.sequence_length]
                    sequences.append(seq.astype(np.float32))

                    # Get entity category
                    entity_info = entities_df[entities_df['entity_id'] == entity_id]
                    if len(entity_info) > 0:
                        category = entity_info.iloc[0]['service_type']
                        categories.append(self.label_encoder.fit_transform([category])[0])

            if not sequences:
                raise ValueError("No valid sequences generated")

            sequences = np.array(sequences, dtype=np.float32)
            categories = np.array(categories, dtype=np.int32)

            logger.info(f"Generated {len(sequences)} sequences with shape {sequences.shape}")
            return sequences, categories

        except Exception as e:
            logger.error(f"Error in prepare_temporal_data: {str(e)}")
            raise

    def save_state(self):
        """Return preprocessor state for saving"""
        return {
            'sequence_length': self.sequence_length,
            'feature_means': self.feature_scaler.mean_.tolist() if hasattr(self.feature_scaler, 'mean_') else None,
            'feature_stds': self.feature_scaler.scale_.tolist() if hasattr(self.feature_scaler, 'scale_') else None,
            'service_type_map': {label: idx for idx, label in enumerate(self.label_encoder.classes_)} if hasattr(self.label_encoder, 'classes_') else None
        }

    def load_state(self, state):
        """Load preprocessor state"""
        if not state:
            logger.warning("Empty state provided, skipping load")
            return

        try:
            self.sequence_length = state['sequence_length']

            if state.get('feature_means') is not None:
                self.feature_scaler.mean_ = np.array(state['feature_means'])
                self.feature_scaler.scale_ = np.array(state['feature_stds'])
                self.feature_scaler.n_samples_seen_ = 1

            if state.get('service_type_map') is not None:
                self.label_encoder.classes_ = np.array(list(state['service_type_map'].keys()))

        except Exception as e:
            logger.error(f"Error loading preprocessor state: {str(e)}")
            raise