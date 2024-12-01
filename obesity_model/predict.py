"""
Example script for making predictions with trained obesity model.
"""

from pathlib import Path
import numpy as np

from .model import Perceptron
from .data_handler import ObesityDataset
from .train import train_obesity_model


def load_or_train_model(model_path: Path) -> Perceptron:
    """Load trained model from file or train a new one if not found"""
    try:
        model = Perceptron()
        model.load(model_path)
        print(f"Model successfully loaded from {model_path}")
    except (FileNotFoundError, KeyError):
        print("\nNo trained model found. Training a new model...")
        model = train_obesity_model(model_path=model_path, quiet=True)
        print("Model training completed and saved.")
    return model


def main():
    """Main prediction script"""
    # Initialize paths
    current_dir = Path(__file__).parent
    model_path = current_dir / 'model_parameters.json'
    
    print("Obesity Prediction Program")
    print("-" * 25)
    
    # Load or train model
    model = load_or_train_model(model_path)
    
    # Test cases
    test_cases = [
        (70, 160),    # 5'10", 160 lbs
        (65, 190),    # 5'5", 190 lbs
        (72, 200),    # 6'0", 200 lbs
        (63, 120),    # 5'3", 120 lbs
        (68, 210),    # 5'8", 210 lbs
    ]
    
    print("\nMaking predictions:")
    print("Height(in) Weight(lbs)  Prediction")
    print("-" * 35)
    
    for height, weight in test_cases:
        # Make prediction
        prediction = model.predict(np.array([[height, weight]]))
        result = ObesityDataset.format_prediction(height, weight, prediction[0])
        
        # Display result
        print(f"{height:^8} {weight:^10}  {result['prediction']}")


if __name__ == '__main__':
    main()
