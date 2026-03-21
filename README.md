# Logistic Regression Model for Heart Disease Prediction

## Project Overview
This project implements a complete machine learning pipeline to predict whether a patient has heart disease using the **Cardiovascular Disease Dataset**. The model uses **Logistic Regression** for binary classification (disease vs. no disease).

## Dataset
- **Name**: Cardiovascular Disease Dataset
- **File**: `Logistic/Cardiovascular_Disease_Dataset.csv`
- **Target Variable**: `target` (1 = Disease, 0 = No Disease)
- **Features**: 13 patient health metrics (age, gender, blood pressure, cholesterol, etc.)

## Machine Learning Pipeline

### 1. Data Understanding & Exploration
- Load and inspect dataset shape and structure
- Display basic statistics and data types
- Identify and remove unwanted ID columns

### 2. Data Cleaning
- Handle missing values (median for numerical, mode for categorical)
- Remove duplicate rows
- Verify data consistency and quality

### 3. Exploratory Data Analysis (EDA)
- Analyze target variable distribution
- Generate correlation heatmap
- Visualize feature distributions
- Identify relationships with target variable

### 4. Feature Selection
- Rank features by correlation with target
- Decision: Keep all features for comprehensive analysis
- Total: 12 features (excluding target)

### 5. Outlier Detection & Handling
- Use IQR (Interquartile Range) method
- Detect outliers in numerical features
- Cap outliers using IQR bounds (winsorization)
- Visualize before/after boxplots

### 6. Data Preprocessing
- **Encoding**: Convert categorical variables using Label Encoding
- **Scaling**: Standardize numerical features with StandardScaler
- Mean: ~0.0, Std: ~1.0 for normalized features

### 7. Train-Test Split
- 80% Training data
- 20% Testing data
- Stratified split to maintain class distribution
- Random state: 42 (reproducibility)

### 8. Model Training
- **Algorithm**: Logistic Regression
- **Solver**: lbfgs (Limited-memory BFGS)
- **Max Iterations**: 1000
- **Parameters**: Random state = 42

### 9. Overfitting Check
- Compare training vs testing accuracy
- Analyze generalization performance
- Detect underfitting/overfitting

### 10. Model Evaluation
Metrics used:
- **Accuracy**: Overall correctness
- **Confusion Matrix**: TP, TN, FP, FN analysis
- **Sensitivity (Recall)**: True Positive Rate
- **Specificity**: True Negative Rate
- **Precision**: Positive Prediction Value
- **F1-Score**: Harmonic mean of Precision & Recall
- **Classification Report**: Per-class performance

### 11. Model Saving
- Trained model: `logistic_regression_model.joblib`
- Scaler: `scaler.joblib`
- Model info: `model_info.txt`

### 12. Prediction on New Data
- Load trained model and scaler
- Preprocess new data
- Generate predictions

## Project Structure
```
ML-Assignment-/
├── Logistic/
│   ├── logistic.ipynb                        # Main notebook with complete ML pipeline
│   ├── Cardiovascular_Disease_Dataset.csv    # Input dataset
│   ├── logistic_regression_model.joblib      # Trained model
│   ├── scaler.joblib                         # Feature scaler
│   ├── model_info.txt                        # Model performance metrics
│   ├── README.md                             # Project documentation
│   └── additional.txt                        # Additional notes
└── README.md                                 # This file
```

## Installation & Requirements

### Required Libraries
```python
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

### Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Navigate to project folder
4. Open Jupyter Notebook: `jupyter notebook logistic.ipynb`

## Model Performance (Latest)
- **Test Accuracy**: Check `model_info.txt` for latest metrics
- **Key Metrics**: Sensitivity, Specificity, Precision, F1-Score included
- **Training Samples**: ~80% of dataset
- **Testing Samples**: ~20% of dataset

## Usage

### Loading & Using the Trained Model
```python
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load('logistic_regression_model.joblib')
scaler = joblib.load('scaler.joblib')

# Prepare new patient data
new_data = pd.DataFrame({...})  # Your patient data

# Preprocess and predict
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
probability = model.predict_proba(new_data_scaled)
```

## Git Branches
- **main**: Main production branch
- **logistic**: Development branch with complete ML implementation
- Other branches: decision-tree, random-forest, svm (alternative models)

## Commits Timeline
- March 5: Folder structure updated
- March 6: Added markdown and dataset
- March 8: Data cleaning and EDA
- March 15: Feature selection and outlier handling
- March 16: Data preprocessing and model training
- March 20: Overfitting check and model evaluation
- March 22: Model saving and finalization

## Author
Nandun Perera

## Contact
Email: nandunperera2001@gmail.com
GitHub: https://github.com/perera2001/ML-Assignment-
