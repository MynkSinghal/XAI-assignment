# Heart Disease Prediction Pipeline

## Project Overview

This project implements a complete machine learning pipeline for predicting heart disease using the UCI Heart Disease dataset (Cleveland subset). The pipeline includes data acquisition, preprocessing, model training, evaluation, explainability analysis using SHAP, and fairness assessment across demographic groups.

**Authors:** Mayank Singhal, Utkarsh Shukla  
**Group:** Group 07  
**Course:** XAI Assignment

## Features

- **Data Pipeline**: Automated fetching and preprocessing of UCI Heart Disease dataset
- **Multiple Models**: Logistic Regression and Random Forest classifiers
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, and ROC-AUC metrics
- **Explainability**: SHAP (SHapley Additive exPlanations) visualizations
- **Fairness Analysis**: Performance evaluation across sex and age groups
- **Rich Visualizations**: Confusion matrices, ROC curves, feature importance plots
- **Reproducibility**: Complete serialization of models and preprocessing metadata

## Project Structure

```
Group07_XAI_Assignment_MayankSinghal_UtkarshShukla/XAI_HeartDisease/
├── data/                       # Processed datasets
│   └── heart_disease.csv       # Cleaned and preprocessed data
├── app/                        # Application resources
│   └── models/                 # Serialized models and metadata
│       ├── logistic_regression_model.pkl
│       ├── random_forest_model.pkl
│       └── preprocessing_metadata.json
├── visuals/                    # Generated visualizations
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── feature_importance.png
│   ├── feature_importance_data.csv
│   ├── shap_summary.png
│   ├── shap_bar.png
│   ├── shap_dependence.png
│   ├── shap_force.png
│   ├── fairness_by_sex.csv
│   ├── fairness_by_age.csv
│   └── metrics.csv
├── notebooks/                  # Jupyter notebooks for exploration
├── report/                     # Analysis reports and narratives
│   └── analytical_narrative.md
├── scripts/                    # Pipeline scripts
│   ├── pipeline.py             # Core pipeline module
│   └── run_pipeline.py         # Pipeline runner script
├── models/                     # Alternative model storage (optional)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Dataset

The project uses the **UCI Heart Disease Dataset** (ID: 45), specifically the Cleveland subset with 14 attributes:

### Features:
- **Numeric Features**: age, trestbps (resting blood pressure), chol (cholesterol), thalach (max heart rate), oldpeak (ST depression)
- **Categorical Features**: sex, cp (chest pain type), fbs (fasting blood sugar), restecg (resting ECG), exang (exercise-induced angina), slope, ca (number of major vessels), thal (thalassemia)

### Target Variable:
- `num`: Original target (0-4 scale)
- `disease_present`: Binarized target (0 = no disease, 1 = disease present)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or navigate to the project directory:**
   ```bash
   cd Group07_XAI_Assignment_MayankSinghal_UtkarshShukla/XAI_HeartDisease
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Complete Pipeline

Execute the pipeline to regenerate all assets from scratch:

```bash
python scripts/run_pipeline.py
```

Or directly run the pipeline module:

```bash
python -m scripts.pipeline
```

### Pipeline Stages

The pipeline executes the following stages:

1. **Directory Setup**: Creates all required project directories
2. **Data Acquisition**: Fetches UCI Heart Disease dataset (id=45)
3. **Data Cleaning**: 
   - Replaces missing values ('?') with NaN
   - Mode-imputes 'ca' and 'thal' columns
   - Binarizes target variable
4. **Feature Engineering**:
   - One-hot encoding of categorical features using `pd.get_dummies`
   - Feature alignment across train/test sets
5. **Data Splitting**: Stratified 80/20 train-test split
6. **Model Training**:
   - Logistic Regression (solver='liblinear', max_iter=1000)
   - Random Forest (n_estimators=200, max_depth=8, random_state=42)
7. **Model Evaluation**: Computes all performance metrics
8. **Visualization Generation**:
   - Confusion matrices
   - ROC curves
   - Feature importance plots
   - SHAP explainability visualizations
9. **Fairness Analysis**: Performance by sex and age groups
10. **Narrative Generation**: 500-700 word analytical report
11. **Artifact Serialization**: Saves models and metadata

### Expected Outputs

After running the pipeline, the following files will be generated:

#### Data Files
- `data/heart_disease.csv`: Processed dataset

#### Models
- `app/models/logistic_regression_model.pkl`: Trained Logistic Regression model
- `app/models/random_forest_model.pkl`: Trained Random Forest model
- `app/models/preprocessing_metadata.json`: Preprocessing configuration and feature names

#### Visualizations
- `visuals/metrics.csv`: Model performance metrics table
- `visuals/confusion_matrix.png`: Confusion matrix for best model
- `visuals/roc_curves.png`: ROC curves for all models
- `visuals/feature_importance.png`: Top feature importances (Random Forest)
- `visuals/feature_importance_data.csv`: Complete feature importance data
- `visuals/shap_summary.png`: SHAP summary beeswarm plot
- `visuals/shap_bar.png`: SHAP mean absolute value bar plot
- `visuals/shap_dependence.png`: SHAP dependence plot for top feature
- `visuals/shap_force.png`: SHAP force plot for representative example
- `visuals/fairness_by_sex.csv`: Performance metrics by sex
- `visuals/fairness_by_age.csv`: Performance metrics by age group

#### Reports
- `report/analytical_narrative.md`: Comprehensive analytical narrative (500-700 words)

## Model Configuration

### Logistic Regression
- **Solver**: liblinear
- **Max Iterations**: 1000
- **Regularization**: L2 (default)

### Random Forest
- **Number of Estimators**: 200
- **Max Depth**: 8
- **Random State**: 42

## Evaluation Metrics

The pipeline computes the following metrics for each model:

- **Accuracy**: Overall correctness
- **Precision**: Proportion of positive predictions that are correct
- **Recall**: Proportion of actual positives correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve

## Explainability

The pipeline uses SHAP (SHapley Additive exPlanations) to provide model interpretability:

- **Summary Plot**: Shows feature importance and effects across all predictions
- **Bar Plot**: Mean absolute SHAP values for feature ranking
- **Dependence Plot**: Relationship between a feature and its SHAP values
- **Force Plot**: Detailed explanation for individual predictions

## Fairness Analysis

Performance is evaluated across demographic groups:

- **By Sex**: Compares metrics between male and female patients
- **By Age**: Analyzes performance across age brackets (<50, 50-60, 60+)

## Reproducibility

All random operations use `random_state=42` for reproducibility. To regenerate identical results:

1. Use the same Python environment (same package versions)
2. Run the pipeline script without modifications
3. All outputs will be deterministic

## Loading Trained Models

To load and use the trained models in other scripts:

```python
import joblib
import json

# Load Random Forest model
model = joblib.load('app/models/random_forest_model.pkl')

# Load preprocessing metadata
with open('app/models/preprocessing_metadata.json', 'r') as f:
    metadata = json.load(f)

feature_names = metadata['feature_names']

# Make predictions
# predictions = model.predict(X_test)
```

## Troubleshooting

### Common Issues

1. **Import Error for ucimlrepo**:
   ```bash
   pip install ucimlrepo
   ```

2. **SHAP Installation Issues**:
   ```bash
   pip install shap --no-cache-dir
   ```

3. **Memory Issues with SHAP**:
   - The pipeline uses a sample size of 100 for SHAP analysis
   - Reduce sample size in `pipeline.py` if needed

4. **Missing Directories**:
   - The pipeline automatically creates all required directories
   - Ensure write permissions in the project directory

## Future Enhancements

- Streamlit web application for interactive predictions
- Additional model algorithms (XGBoost, Neural Networks)
- Hyperparameter tuning with cross-validation
- Extended fairness metrics using fairlearn library
- Real-time model monitoring and retraining pipeline

## Contributing

This project is part of an academic assignment. For questions or improvements:

- Contact: Mayank Singhal, Utkarsh Shukla
- Course: XAI Assignment

## License

This project is for educational purposes as part of the XAI assignment.

## References

- UCI Machine Learning Repository: Heart Disease Dataset
- Scikit-learn Documentation: https://scikit-learn.org/
- SHAP Documentation: https://shap.readthedocs.io/
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. NeurIPS.

## Acknowledgments

- UCI Machine Learning Repository for providing the Heart Disease dataset
- The SHAP library developers for explainability tools
- Course instructors for guidance on XAI techniques
