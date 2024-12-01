# ML-Math Project

A simple machine learning project implementing a perceptron algorithm to classify obesity based on height and weight measurements.

## Project Structure
- `perceptron.py`: Main implementation of the perceptron algorithm
- `predict_example.py`: Example script showing how to use the trained model
- `data.csv`: Training data with height, weight, and obesity classifications
- `requirements.txt`: Python dependencies

## Setup
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

## Usage
1. Train the model:
```bash
python perceptron.py
```

2. Make predictions:
```bash
python predict_example.py
```

## Model Details
The perceptron uses two features:
- Height (inches)
- Weight (pounds)

It classifies individuals as:
- 1: Obese
- -1: Not Obese

The model parameters are saved after training, so you don't need to retrain the model each time you want to make predictions.
