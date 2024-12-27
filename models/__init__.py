"""
TimeGNN Models
-------------
Core model implementations for temporal and structural anomaly detection.
"""

from .time_gnn_model import TimeGNNAnomalyDetector
from .structural_gnn_model import StructuralGNNDetector

__all__ = [
    'TimeGNNAnomalyDetector',
    'StructuralGNNDetector'
]