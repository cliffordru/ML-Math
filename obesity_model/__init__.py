"""
Obesity classification package using perceptron model.
"""

from .model import Perceptron, BinaryClassifier
from .data_handler import ObesityDataset
from .train import train_obesity_model, evaluate_model

__all__ = [
    'Perceptron',
    'BinaryClassifier',
    'ObesityDataset',
    'train_obesity_model',
    'evaluate_model'
]
