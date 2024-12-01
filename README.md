# ML-Math Project

[GrayTechnology.com/blog](https://graytechnology.com/blog) - Need an AI, machine learning or data architect / engineer or technology leader? I'm available - info@graytechnology.com

A collection of machine learning implementations for mathematical concepts.

## Projects

### 1. Obesity Model
A perceptron-based binary classifier that predicts obesity status based on height and weight measurements. The implementation follows SOLID principles and Python best practices.

#### Architecture
The project is organized into modular components following clean architecture principles:

```
obesity_model/
├── __init__.py           # Package initialization and public interfaces
├── model.py             # Core model implementations and abstractions
├── data_handler.py      # Data loading and processing utilities
├── train.py            # Model training functionality
├── predict.py          # Prediction interface and examples
├── data.csv            # Training data
└── model_parameters.json # Trained model weights (created during training)
```

#### Key Components
- **Model (model.py)**
  - `BinaryClassifier`: Abstract base class for classification models
  - `Perceptron`: Implementation of perceptron algorithm
  - `DataNormalizer`: Protocol for data normalization strategies
  - `StandardScaler`: Implementation of standardization normalization

- **Data Handling (data_handler.py)**
  - `ObesityDataset`: Class for loading and processing obesity data
  - Data validation and formatting utilities

- **Training (train.py)**
  - Model training functionality
  - Performance evaluation
  - Model persistence

- **Prediction (predict.py)**
  - Example prediction interface
  - Test cases and formatting

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

You can start making predictions right away:
```bash
.\venv\Scripts\python.exe -m obesity_model.predict
```
This will:
- Load the trained model if available, or automatically train a new one if not found
- Run predictions on several test cases:
  * 5'10" (70"), 160 lbs
  * 5'5" (65"), 190 lbs
  * 6'0" (72"), 200 lbs
  * 5'3" (63"), 120 lbs
  * 5'8" (68"), 210 lbs
- Display formatted results

If you want to retrain the model manually:
```bash
.\venv\Scripts\python.exe -m obesity_model.train
```
This will:
- Load the training data from `data.csv`
- Train the perceptron model
- Save the model parameters to `model_parameters.json`
- Display training progress and final accuracy

#### Model Details
The obesity classifier uses:

**Input Features:**
- Height (inches)
- Weight (pounds)

**Classification:**
- 1: Obese
- -1: Not Obese

**Technical Features:**
- Automatic feature normalization
- Configurable learning rate and epochs
- Early stopping on perfect classification
- Type hints and comprehensive documentation
- Modular and extensible design

#### Development
The codebase follows:
- Python type hints
- Comprehensive documentation
- Clean architecture patterns
- Modern Python practices (pathlib, dataclasses)

To extend the model:
1. Implement new classifiers by inheriting from `BinaryClassifier`
2. Add new normalizers by implementing the `DataNormalizer` protocol
3. Extend data handling by modifying `ObesityDataset`
