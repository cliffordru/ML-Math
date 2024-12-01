"""
Data handling utilities for obesity classification.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict
import pandas as pd
import numpy as np


@dataclass
class ObesityDataset:
    """Class for handling obesity classification dataset"""
    
    data_path: Path
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and prepare the dataset
        
        Returns:
            Tuple of (features, labels) where:
            - features is a numpy array of shape (n_samples, 2) containing height and weight
            - labels is a numpy array of shape (n_samples,) containing obesity labels (-1 or 1)
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        data = pd.read_csv(self.data_path)
        
        # Extract features and labels
        X = data[['height_in', 'weight_lbs']].values
        y = data['obese'].values
        
        return X, y
    
    @staticmethod
    def format_prediction(height: float, weight: float, prediction: int) -> Dict[str, str]:
        """
        Format a prediction result
        
        Args:
            height: Height in inches
            weight: Weight in pounds
            prediction: Model prediction (-1 or 1)
            
        Returns:
            Dictionary containing formatted prediction information
        """
        return {
            'height': f"{height} inches",
            'weight': f"{weight} lbs",
            'prediction': "Obese" if prediction == 1 else "Not Obese"
        }
