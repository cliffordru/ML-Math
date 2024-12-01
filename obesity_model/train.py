"""
Training script for obesity classification model.
"""

from pathlib import Path
from typing import Optional
import numpy as np

from .model import Perceptron
from .data_handler import ObesityDataset


def train_obesity_model(
    data_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    learning_rate: float = 0.01,
    max_epochs: int = 100,
    quiet: bool = False
) -> Perceptron:
    """
    Train the obesity classification model
    
    Args:
        data_path: Path to training data CSV file
        model_path: Path to save trained model
        learning_rate: Learning rate for training
        max_epochs: Maximum number of training epochs
        quiet: If True, suppress training progress messages
        
    Returns:
        Trained Perceptron model
    """
    # Set default paths relative to this file
    if data_path is None:
        data_path = Path(__file__).parent / 'data.csv'
    if model_path is None:
        model_path = Path(__file__).parent / 'model_parameters.json'
    
    # Load and prepare data
    dataset = ObesityDataset(data_path)
    X, y = dataset.load_data()
    
    # Initialize and train model
    model = Perceptron(
        learning_rate=learning_rate,
        max_epochs=max_epochs
    )
    
    if not quiet:
        print("\nTraining the model...")
        print(f"Initial weights: {model.weights}")
        print(f"Initial bias: {model.bias}")
    
    model.train(X, y)
    
    # Save trained model
    model.save(model_path)
    if not quiet:
        print(f"Model successfully saved to {model_path}")
    
    return model


def evaluate_model(model: Perceptron, X: np.ndarray, y: np.ndarray, quiet: bool = False) -> float:
    """
    Evaluate model performance
    
    Args:
        model: Trained Perceptron model
        X: Feature matrix
        y: True labels
        quiet: If True, suppress evaluation messages
        
    Returns:
        Accuracy score (0 to 1)
    """
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    if not quiet:
        print(f"\nModel Accuracy: {accuracy:.2%}")
    return accuracy


if __name__ == '__main__':
    # Train the model
    model = train_obesity_model()
    
    # Load data for evaluation
    dataset = ObesityDataset(Path(__file__).parent / 'data.csv')
    X, y = dataset.load_data()
    
    # Evaluate the model
    evaluate_model(model, X, y)
    
    # Example predictions
    print("\nExample predictions:")
    test_cases = [
        (70, 160),  # 5'10", 160 lbs
        (65, 190),  # 5'5", 190 lbs
    ]
    
    for height, weight in test_cases:
        prediction = model.predict(np.array([[height, weight]]))
        result = ObesityDataset.format_prediction(height, weight, prediction[0])
        print(f"Height: {result['height']}, Weight: {result['weight']} -> {result['prediction']}")
