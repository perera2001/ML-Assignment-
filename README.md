# Heart Disease Prediction Models - Comparative Study

## Project Overview
This project implements three different machine learning pipelines to predict whether a patient has heart disease using the **Cardiovascular Disease Dataset**. The project compares three popular classification algorithms:
- **Decision Tree**
- **Logistic Regression**
- **Random Forest**

Each branch contains a complete ML implementation for binary classification (disease vs. no disease).

## Dataset
- **Name**: Cardiovascular Disease Dataset
- **Target Variable**: `target` (1 = Disease, 0 = No Disease)
- **Features**: 13 patient health metrics (age, gender, blood pressure, cholesterol, etc.)
- **Location**: Available in each branch's folder

## Common Machine Learning Pipeline

All three models follow this standardized pipeline:

1. **Data Understanding & Exploration**: Load, inspect, and analyze dataset structure
2. **Data Cleaning**: Handle missing values, remove duplicates, ensure data quality
3. **Exploratory Data Analysis (EDA)**: Analyze distributions, correlations, and relationships
4. **Feature Selection**: Rank and select important features
5. **Outlier Detection & Handling**: Use IQR method to detect and handle outliers
6. **Data Preprocessing**: Encode categorical variables and scale numerical features
7. **Train-Test Split**: 80% training, 20% testing with stratification
8. **Model Training**: Train the specific algorithm
9. **Overfitting Check**: Compare training vs testing performance
10. **Model Evaluation**: Calculate metrics (Accuracy, Precision, Recall, F1-Score, etc.)
11. **Feature Importance Analysis**: Understand which features drive predictions
12. **Model Saving**: Save trained model and preprocessing objects
13. **Prediction on New Data**: Generate predictions on unseen data

## Branch Details

### 1. Decision Tree Branch
**Location**: `decision-tree` branch / `Decision tree/` folder

**Model**: Decision Tree Classifier
- **File**: `decisiontree.ipynb`
- **Saved Models**:
  - `heart_disease_decision_tree_model.pkl`
  - `heart_disease_decision_tree_model.joblib`
  - `feature_names.joblib`

**Key Characteristics**:
- Interpretable tree-based model
- No scaling required
- Good for capturing non-linear relationships
- Prone to overfitting (large trees)

**Use Case**: When interpretability and understanding decision paths are important

---

### 2. Logistic Regression Branch
**Location**: `logistic` branch / `Logistic/` folder

**Model**: Logistic Regression
- **File**: `logistic.ipynb`
- **Saved Models**:
  - `logistic_regression_model.joblib`
  - `scaler.joblib`
  - `model_info.txt`

**Key Characteristics**:
- Linear classification model
- Requires feature scaling
- Fast training and inference
- Good baseline model
- Provides probability estimates

**Use Case**: Fast, interpretable probabilistic predictions

---

### 3. Random Forest Branch
**Location**: `random-forest` branch / `random forest/` folder

**Model**: Random Forest Classifier
- **File**: `random_forest.ipynb` (or similar)
- **Key Characteristics**:
  - Ensemble of multiple decision trees
  - Reduces overfitting through averaging
  - Captures complex patterns
  - Provides feature importance scores
  - No scaling required

**Use Case**: When high accuracy is needed with reduced overfitting

## Project Structure
```
ML-Assignment-/
├── Decision tree/
│   ├── decisiontree.ipynb                        # Decision Tree ML pipeline
│   ├── heart_disease_decision_tree_model.pkl     # Trained model (pickle format)
│   ├── heart_disease_decision_tree_model.joblib  # Trained model (joblib format)
│   ├── feature_names.joblib                      # Feature names
│   └── Cardiovascular_Disease_Dataset.csv        # Input dataset
├── Logistic/
│   ├── logistic.ipynb                            # Logistic Regression ML pipeline
│   ├── Cardiovascular_Disease_Dataset.csv        # Input dataset
│   ├── logistic_regression_model.joblib          # Trained model
│   ├── scaler.joblib                             # Feature scaler
│   ├── model_info.txt                            # Model performance metrics
│   └── README.md                                 # Detailed documentation
├── random forest/
│   ├── random_forest.ipynb                       # Random Forest ML pipeline
│   ├── random_forest_model.joblib                # Trained model
│   ├── Cardiovascular_Disease_Dataset.csv        # Input dataset
│   └── feature_importance.joblib                 # Feature importance scores
└── README.md                                     # This file
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

## Model Performance Summary

### Evaluation Metrics (Common to All Models)
- **Accuracy**: Overall correctness of predictions
- **Confusion Matrix**: TP, TN, FP, FN analysis
- **Sensitivity (Recall)**: True Positive Rate (Disease detection rate)
- **Specificity**: True Negative Rate (Healthy detection rate)
- **Precision**: Positive Prediction Value
- **F1-Score**: Harmonic mean of Precision & Recall

Check each branch's documentation for specific model performance metrics.

### Model Comparison
| Metric | Decision Tree | Logistic Regression | Random Forest |
|--------|---------------|-------------------|---------------|
| Training | Fast | Fast | Medium |
| Prediction | Very Fast | Very Fast | Fast |
| Interpretability | High | High | Medium |
| Scalability | Good | Excellent | Good |
| Overfitting Risk | High | Low | Low |

## Usage

### Loading & Using the Decision Tree Model
```python
import joblib
import pandas as pd

# Load model and feature names
model = joblib.load('Decision tree/heart_disease_decision_tree_model.joblib')
feature_names = joblib.load('Decision tree/feature_names.joblib')

# Prepare new patient data
new_data = pd.DataFrame({...})  # Your patient data

# Predict (no scaling needed)
prediction = model.predict(new_data)
probability = model.predict_proba(new_data)
```

### Loading & Using the Logistic Regression Model
```python
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load('Logistic/logistic_regression_model.joblib')
scaler = joblib.load('Logistic/scaler.joblib')

# Prepare and scale new patient data
new_data = pd.DataFrame({...})  # Your patient data
new_data_scaled = scaler.transform(new_data)

# Predict
prediction = model.predict(new_data_scaled)
probability = model.predict_proba(new_data_scaled)
```

### Loading & Using the Random Forest Model
```python
import joblib
import pandas as pd

# Load model
model = joblib.load('random forest/random_forest_model.joblib')

# Prepare new patient data
new_data = pd.DataFrame({...})  # Your patient data

# Predict (no scaling needed)
prediction = model.predict(new_data)
probability = model.predict_proba(new_data)

# Get feature importance
feature_importance = model.feature_importances_
```

## Git Branches

| Branch | Model | Description |
|--------|-------|-------------|
| `main` | - | Main production branch |
| `decision-tree` | Decision Tree | Tree-based classifier for interpretable predictions |
| `logistic` | Logistic Regression | Linear classifier for baseline probabilistic model |
| `random-forest` | Random Forest | Ensemble model for improved accuracy and robustness |

Each branch contains:
- Complete ML pipeline in a Jupyter notebook
- Trained model artifacts
- Feature preprocessing information
- Commit history showing development stages

## Development Timeline

### Decision Tree Branch (Current)
- March 3: Dataset added
- March 3: Import libraries
- March 3: Import packages
- March 8: Load dataset and inspect basic structure
- March 8: Clean dataset and handle missing values
- March 9: Add exploratory data analysis and summary statistics
- March 10: Feature selection
- March 11: Outlier detection
- March 13: Data preprocessing
- March 15: Train test splitting
- March 16: Train test splitting updated
- March 16: Model training
- March 17: Model visualizing
- March 19: Feature importance analysis
- March 20: Overfitting check
- March 20: Model evaluation
- March 21: Model saving
- March 23: Prediction on new data
- March 23: Summary and key takeaways added
- March 23: Model outputs

### Other Branches
- **logistic**: Complete ML pipeline with Logistic Regression
- **random-forest**: Complete ML pipeline with Random Forest
- **main**: Stable production branch

## Author
Nandun Perera

## Contact
Email: nandunperera2001@gmail.com
GitHub: https://github.com/perera2001/ML-Assignment-
