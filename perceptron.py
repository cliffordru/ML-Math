import numpy as np
import pandas as pd
import json

# Global variables to store model parameters
w1 = w2 = b = 0
data_mean = data_std = None

def train_model():
    global w1, w2, b, data_mean, data_std
    # Load and prepare the data
    data = pd.read_csv('data.csv')

    # Normalize the features (height and weight) to help with convergence
    X = data[['height_in', 'weight_lbs']].values
    data_mean = X.mean(axis=0)
    data_std = X.std(axis=0)
    X = (X - data_mean) / data_std  # standardize features
    y = data['obese'].values

    # Initialize weights and bias randomly
    np.random.seed(42)  # for reproducibility
    w1, w2 = np.random.randn(2) * 0.01  # small random numbers
    b = 0

    # Training parameters
    learning_rate = 0.01
    epochs = 100

    # Training loop
    print("Training the model...")
    print("Initial weights: w1 = {:.4f}, w2 = {:.4f}, b = {:.4f}".format(w1, w2, b))

    for epoch in range(epochs):
        total_error = 0
        
        # Train on each data point
        for i in range(len(X)):
            # Calculate prediction
            x1, x2 = X[i]
            sum_value = w1 * x1 + w2 * x2 + b
            prediction = 1 if sum_value > 0 else -1
            
            # Update weights and bias if prediction is wrong
            error = y[i] - prediction
            total_error += abs(error)
            
            if error != 0:
                w1 += learning_rate * error * x1
                w2 += learning_rate * error * x2
                b += learning_rate * error
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Total Error: {total_error}")
        
        # Stop if perfect classification is achieved
        if total_error == 0:
            print(f"\nPerfect classification achieved at epoch {epoch + 1}!")
            break

    print("\nFinal weights: w1 = {:.4f}, w2 = {:.4f}, b = {:.4f}".format(w1, w2, b))
    
    # Save the model after training
    save_model()
    return True

def predict_obesity(height, weight):
    """
    Predict if a person is obese based on their height and weight.
    
    Args:
        height (float): Height in inches
        weight (float): Weight in pounds
    
    Returns:
        int: 1 if predicted obese, -1 if predicted not obese
    """
    # Normalize input using same parameters as training data
    height_norm = (height - data_mean[0]) / data_std[0]
    weight_norm = (weight - data_mean[1]) / data_std[1]
    
    sum_value = w1 * height_norm + w2 * weight_norm + b
    return 1 if sum_value > 0 else -1

def test_model():
    # Test the model on the training data
    data = pd.read_csv('data.csv')
    X = data[['height_in', 'weight_lbs']].values
    X = (X - data_mean) / data_std  # standardize features
    y = data['obese'].values

    correct = 0
    print("\nTesting the model on training data:")
    print("Height(in) Weight(lbs) Actual Predicted")
    print("-" * 40)

    for i in range(len(X)):
        x1, x2 = X[i]
        sum_value = w1 * x1 + w2 * x2 + b
        prediction = 1 if sum_value > 0 else -1
        
        # Use original (non-normalized) values for display
        height = data['height_in'].values[i]
        weight = data['weight_lbs'].values[i]
        
        correct += (prediction == y[i])
        print(f"{height:8.0f} {weight:10.0f} {y[i]:7d} {prediction:9d}")

    accuracy = correct / len(X) * 100
    print(f"\nAccuracy: {accuracy:.2f}%")

def save_model(filename='model_parameters.json'):
    """Save the model parameters to a file"""
    global w1, w2, b, data_mean, data_std
    
    # Check if model parameters exist
    if data_mean is None or data_std is None:
        print("Error: Model parameters not initialized. Please train the model first.")
        return False
        
    try:
        model_params = {
            'w1': w1,
            'w2': w2,
            'b': b,
            'data_mean': data_mean.tolist(),
            'data_std': data_std.tolist()
        }
        with open(filename, 'w') as f:
            json.dump(model_params, f)
        print(f"Model successfully saved to {filename}")
        print(f"Parameters: w1={w1:.4f}, w2={w2:.4f}, b={b:.4f}")
        return True
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        return False

def load_model(filename='model_parameters.json'):
    """Load the model parameters from a file"""
    global w1, w2, b, data_mean, data_std
    try:
        with open(filename, 'r') as f:
            model_params = json.load(f)
        w1 = model_params['w1']
        w2 = model_params['w2']
        b = model_params['b']
        data_mean = np.array(model_params['data_mean'])
        data_std = np.array(model_params['data_std'])
        return True
    except FileNotFoundError:
        print(f"No saved model found at {filename}")
        return False

if __name__ == '__main__':
    # Train the model when the script is run directly
    print("Starting training process...")
    train_model()
    test_model()
    
    # Example predictions
    print("\nExample predictions:")
    test_cases = [
        (70, 160),  # should be non-obese
        (65, 190),  # should be obese
    ]

    for height, weight in test_cases:
        prediction = predict_obesity(height, weight)
        status = "Obese" if prediction == 1 else "Not Obese"
        print(f"Height: {height}in, Weight: {weight}lbs -> {status}")