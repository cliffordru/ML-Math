"""
Base model interface and implementations for classification models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, List, Dict, Any, Optional, Tuple
import numpy as np
import json
from pathlib import Path


class DataNormalizer(Protocol):
    """Protocol for data normalization strategies"""
    
    def fit(self, data: np.ndarray) -> None:
        """Learn normalization parameters from data"""
        ...
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply normalization to data"""
        ...
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reverse the normalization"""
        ...

    def to_dict(self) -> Dict[str, Any]:
        """Convert normalizer parameters to dictionary"""
        ...
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataNormalizer':
        """Create normalizer from dictionary parameters"""
        ...


@dataclass
class StandardScaler:
    """Standardization normalization (zero mean, unit variance)"""
    
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    
    def fit(self, data: np.ndarray) -> None:
        """Learn mean and standard deviation from data"""
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Standardize data using learned parameters"""
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fit before transform")
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reverse standardization"""
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fit before inverse_transform")
        return data * self.std + self.mean
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert scaler parameters to dictionary"""
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fit before serialization")
        return {
            'mean': self.mean.tolist(),
            'std': self.std.tolist()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StandardScaler':
        """Create scaler from dictionary parameters"""
        return cls(
            mean=np.array(data['mean']),
            std=np.array(data['std'])
        )


class BinaryClassifier(ABC):
    """Abstract base class for binary classifiers"""
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on data"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on data"""
        pass
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model parameters"""
        pass
    
    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model parameters"""
        pass


@dataclass
class Perceptron(BinaryClassifier):
    """Perceptron implementation for binary classification"""
    
    weights: Optional[np.ndarray] = None
    bias: float = 0.0
    learning_rate: float = 0.01
    max_epochs: int = 100
    random_seed: int = 42
    normalizer: DataNormalizer = field(default_factory=StandardScaler)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the perceptron model
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,) with values in {-1, 1}
        """
        # Initialize weights
        np.random.seed(self.random_seed)
        n_features = X.shape[1]
        self.weights = np.random.randn(n_features) * 0.01
        
        # Fit and apply normalization
        self.normalizer.fit(X)
        X_norm = self.normalizer.transform(X)
        
        print("Training the model...")
        print(f"Initial weights: {self.weights}")
        print(f"Initial bias: {self.bias}")
        
        # Training loop
        for epoch in range(self.max_epochs):
            total_error = 0
            
            for i in range(len(X_norm)):
                # Make prediction
                prediction = self._predict_single(X_norm[i])
                
                # Update weights if prediction is wrong
                error = y[i] - prediction
                total_error += abs(error)
                
                if error != 0:
                    self.weights += self.learning_rate * error * X_norm[i]
                    self.bias += self.learning_rate * error
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}, Total Error: {total_error}")
            
            # Early stopping if perfect classification
            if total_error == 0:
                print(f"\nPerfect classification achieved at epoch {epoch + 1}!")
                break
        
        print(f"\nFinal weights: {self.weights}")
        print(f"Final bias: {self.bias}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples,) with values in {-1, 1}
        """
        if self.weights is None:
            raise ValueError("Model must be trained before prediction")
        
        X_norm = self.normalizer.transform(X)
        return np.array([self._predict_single(x) for x in X_norm])
    
    def _predict_single(self, x: np.ndarray) -> int:
        """Make prediction for a single sample"""
        if self.weights is None:
            raise ValueError("Model must be trained before prediction")
        
        activation = np.dot(x, self.weights) + self.bias
        return 1 if activation > 0 else -1
    
    def save(self, path: Path) -> None:
        """Save model parameters to file"""
        if self.weights is None:
            raise ValueError("Model must be trained before saving")
        
        model_params = {
            'weights': self.weights.tolist(),
            'bias': self.bias,
            'normalizer': self.normalizer.to_dict()
        }
        
        with open(path, 'w') as f:
            json.dump(model_params, f)
        print(f"Model successfully saved to {path}")
    
    def load(self, path: Path) -> None:
        """Load model parameters from file"""
        with open(path, 'r') as f:
            model_params = json.load(f)
        
        self.weights = np.array(model_params['weights'])
        self.bias = model_params['bias']
        self.normalizer = StandardScaler.from_dict(model_params['normalizer'])
