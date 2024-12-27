"""
Hybrid GNN Model Implementation
Combines TimeGNN and StructuralGNN for comprehensive anomaly detection
"""

import logging
import os
import json
import numpy as np
import tensorflow as tf
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
from .time_gnn_model import TimeGNNAnomalyDetector
from .structural_gnn_model import StructuralGNNDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HybridGNNDetector:
    """
    Hybrid Anomaly Detection using both TimeGNN and StructuralGNN models.
    Combines temporal and structural patterns for more robust detection.
    """

    def __init__(self, 
                time_gnn_params: Optional[Dict[str, Any]] = None,
                structural_gnn_params: Optional[Dict[str, Any]] = None,
                temporal_weight: float = 0.6):
        """
        Initialize hybrid detector with both models.
        
        Args:
            time_gnn_params: Parameters for TimeGNN initialization
            structural_gnn_params: Parameters for StructuralGNN initialization
            temporal_weight: Weight for temporal predictions (1 - weight for structural)
        """
        try:
            self.temporal_weight = temporal_weight
            self.structural_weight = 1.0 - temporal_weight
            
            # Initialize individual models
            self.time_gnn = TimeGNNAnomalyDetector(**(time_gnn_params or {}))
            self.structural_gnn = StructuralGNNDetector(**(structural_gnn_params or {}))
            
            self.is_trained = False
            self.training_history: Dict[str, List[float]] = {
                'temporal_loss': [],
                'structural_loss': [],
                'combined_loss': [],
                'val_temporal_loss': [],
                'val_structural_loss': [],
                'val_combined_loss': [],
                'training_time': []
            }
            
            logger.info(
                f"Initialized HybridGNNDetector with weights - "
                f"Temporal: {self.temporal_weight:.2f}, "
                f"Structural: {self.structural_weight:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Error initializing HybridGNNDetector: {str(e)}")
            raise

    def train(self,
             timeseries_df: pd.DataFrame,
             relationships_df: pd.DataFrame,
             epochs: int = 10,
             threshold_percentile: float = 95,
             validation_split: float = 0.2,
             batch_size: int = 32) -> Dict[str, Any]:
        """
        Train both models and combine their results.
        
        Args:
            timeseries_df: Time series data
            relationships_df: Relationship data
            epochs: Number of training epochs
            threshold_percentile: Anomaly threshold percentile
            validation_split: Validation data fraction
            batch_size: Training batch size
            
        Returns:
            Combined training metrics
        """
        try:
            logger.info("Starting hybrid model training...")
            start_time = time.time()
            
            # Train TimeGNN
            logger.info("Training temporal model...")
            temporal_history = self.time_gnn.train(
                timeseries_df=timeseries_df,
                relationships_df=relationships_df,
                epochs=epochs,
                threshold_percentile=threshold_percentile,
                validation_split=validation_split,
                batch_size=batch_size
            )
            
            # Train StructuralGNN
            logger.info("Training structural model...")
            structural_history = self.structural_gnn.train(
                relationships_df=relationships_df,
                epochs=epochs,
                threshold_percentile=threshold_percentile,
                validation_split=validation_split,
                batch_size=batch_size
            )
            
            # Combine training histories
            self.training_history['temporal_loss'].extend(temporal_history['history']['loss'])
            self.training_history['structural_loss'].extend(structural_history['history']['loss'])
            self.training_history['val_temporal_loss'].extend(temporal_history['history']['val_loss'])
            self.training_history['val_structural_loss'].extend(structural_history['history']['val_loss'])
            
            # Calculate combined losses
            combined_loss = [
                t * self.temporal_weight + s * self.structural_weight
                for t, s in zip(
                    self.training_history['temporal_loss'],
                    self.training_history['structural_loss']
                )
            ]
            val_combined_loss = [
                t * self.temporal_weight + s * self.structural_weight
                for t, s in zip(
                    self.training_history['val_temporal_loss'],
                    self.training_history['val_structural_loss']
                )
            ]
            
            self.training_history['combined_loss'] = combined_loss
            self.training_history['val_combined_loss'] = val_combined_loss
            
            training_time = time.time() - start_time
            self.training_history['training_time'].append(training_time)
            
            self.is_trained = True
            
            logger.info(f"Hybrid training completed in {training_time:.2f} seconds")
            logger.info(f"Final combined loss: {combined_loss[-1]:.4f}")
            logger.info(f"Final combined validation loss: {val_combined_loss[-1]:.4f}")
            
            return {
                'history': self.training_history,
                'training_time': training_time,
                'final_temporal_loss': temporal_history['final_loss'],
                'final_structural_loss': structural_history['final_loss'],
                'final_combined_loss': combined_loss[-1]
            }
            
        except Exception as e:
            logger.error(f"Error during hybrid model training: {str(e)}")
            raise

    def predict(self, 
               timeseries_df: pd.DataFrame,
               relationships_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Generate predictions using both models and combine results.
        
        Args:
            timeseries_df: Time series data
            relationships_df: Relationship data
            
        Returns:
            Tuple of (anomaly flags, pattern types, detailed scores)
        """
        try:
            if not self.is_trained:
                raise ValueError("Models must be trained before prediction")
                
            logger.info("Starting hybrid anomaly detection...")
            prediction_start = time.time()
            
            # Get temporal predictions
            temporal_anomalies, temporal_patterns, temporal_scores = self.time_gnn.predict(
                timeseries_df,
                relationships_df
            )
            
            # Get structural predictions
            structural_anomalies, structural_patterns, structural_scores = self.structural_gnn.predict(
                relationships_df
            )
            
            # Combine predictions with weighted voting
            combined_scores = (
                temporal_scores * self.temporal_weight +
                structural_scores * self.structural_weight
            )
            
            # Determine final anomalies using combined scores
            combined_threshold = np.percentile(combined_scores, 95)
            final_anomalies = combined_scores > combined_threshold
            
            # Combine pattern types
            final_patterns = np.where(
                temporal_scores > structural_scores,
                temporal_patterns,
                structural_patterns
            )
            
            prediction_time = time.time() - prediction_start
            n_anomalies = int(np.sum(final_anomalies))
            
            logger.info(f"Hybrid detection completed in {prediction_time:.2f} seconds")
            logger.info(f"Found {n_anomalies} anomalies "
                       f"({(n_anomalies/len(final_anomalies))*100:.2f}% of total)")
            
            if n_anomalies > 0:
                pattern_counts = np.unique(final_patterns[final_anomalies], return_counts=True)
                logger.info("Pattern distribution:")
                for pattern, count in zip(*pattern_counts):
                    logger.info(f"- {pattern}: {count}")
                    
            detailed_scores = {
                'temporal_scores': temporal_scores,
                'structural_scores': structural_scores,
                'combined_scores': combined_scores
            }
            
            return final_anomalies, final_patterns, detailed_scores
            
        except Exception as e:
            logger.error(f"Error during hybrid prediction: {str(e)}")
            raise

    def save_model(self, path: str) -> bool:
        """
        Save both models and hybrid parameters.
        
        Args:
            path: Base path for saving models
        """
        try:
            if not self.is_trained:
                raise ValueError("Cannot save untrained models")
                
            # Create directory structure
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save individual models
            temporal_path = f"{path}_temporal"
            structural_path = f"{path}_structural"
            
            self.time_gnn.save_model(temporal_path)
            self.structural_gnn.save_model(structural_path)
            
            # Save hybrid parameters
            properties = {
                'temporal_weight': self.temporal_weight,
                'structural_weight': self.structural_weight,
                'is_trained': self.is_trained,
                'training_history': self.training_history
            }
            
            properties_path = f"{path}_hybrid_properties.json"
            with open(properties_path, 'w') as f:
                json.dump(properties, f, indent=4)
                
            logger.info(f"Hybrid model saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving hybrid model: {str(e)}")
            raise

    def load_model(self, path: str) -> bool:
        """
        Load both models and hybrid parameters.
        
        Args:
            path: Base path for loading models
        """
        try:
            if not os.path.exists(f"{path}_hybrid_properties.json"):
                raise ValueError(f"Model path {path} does not exist")
                
            # Load individual models
            temporal_path = f"{path}_temporal"
            structural_path = f"{path}_structural"
            
            self.time_gnn.load_model(temporal_path)
            self.structural_gnn.load_model(structural_path)
            
            # Load hybrid parameters
            properties_path = f"{path}_hybrid_properties.json"
            with open(properties_path, 'r') as f:
                properties = json.load(f)
                
            self.temporal_weight = properties['temporal_weight']
            self.structural_weight = properties['structural_weight']
            self.is_trained = properties['is_trained']
            self.training_history = properties['training_history']
            
            logger.info(f"Hybrid model loaded from {path}")
            logger.info(f"Model training state: {'trained' if self.is_trained else 'untrained'}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading hybrid model: {str(e)}")
            raise
