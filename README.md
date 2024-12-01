# ML-Math Project

A collection of machine learning implementations for mathematical concepts.

## Projects

### 1. Obesity Model
A simple perceptron implementation that classifies obesity based on height and weight measurements.

#### Structure
```
obesity_model/
├── __init__.py           # Package initialization
├── perceptron.py         # Core perceptron implementation
├── predict_example.py    # Example usage script
├── data.csv              # Training data
└── model_parameters.json # Trained model weights and parameters - will be created during training
```

#### Setup
1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/MacOS
.\venv\Scripts\activate   # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

#### Usage
1. Train the model:
```bash
python -m obesity_model.perceptron
```
This will create `model_parameters.json` in the obesity_model directory.

2. Make predictions:
```bash
python -m obesity_model.predict_example
```
This will load the saved model parameters and make predictions on sample data.

#### Model Details
The obesity classifier uses two features:
- Height (inches)
- Weight (pounds)

Classification:
- 1: Obese
- -1: Not Obese

The model parameters are saved after training in `obesity_model/model_parameters.json`, so you don't need to retrain the model each time you want to make predictions.
