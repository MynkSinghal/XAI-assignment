# Explainable AI Heart Disease Prediction App

A Streamlit-based web application for heart disease prediction using machine learning models with built-in explainability features.

## Features

- **Dark-themed UI** with responsive design
- **Dual input modes**:
  - Manual patient data entry via form
  - CSV batch upload for multiple predictions
- **Multiple ML models**: Logistic Regression and Random Forest
- **Prediction outputs**: Binary predictions with confidence scores
- **Ensemble recommendations**: Combined insights from both models
- **Model interpretability**:
  - SHAP (SHapley Additive exPlanations) visualizations
  - Feature importance plots
  - Comprehensive metrics display
- **Fairness analysis**: Performance breakdown by demographics
- **Error handling**: Graceful handling of missing assets

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Install Dependencies

```bash
pip install streamlit pandas numpy scikit-learn joblib pillow
```

Or install from requirements file:

```bash
cd Group07_XAI_Assignment_MayankSinghal_UtkarshShukla/XAI_HeartDisease
pip install -r requirements.txt
```

## Running the App

From the project root directory:

```bash
streamlit run XAI_HeartDisease/app/streamlit_app.py
```

The app will open in your default browser at `http://localhost:8501`

## Usage

### Manual Input Mode

1. Navigate to the "Make Predictions" tab
2. Select "Manual Entry" input mode
3. Fill in all 13 clinical features using the form widgets:
   - **Age**: Slider (29-77 years)
   - **Sex**: Select box (Female/Male)
   - **Chest Pain Type**: Select box (1-4)
   - **Resting Blood Pressure**: Slider (94-200 mm Hg)
   - **Serum Cholesterol**: Slider (126-564 mg/dl)
   - **Fasting Blood Sugar**: Select box (Yes/No)
   - **Resting ECG Results**: Select box (0-2)
   - **Maximum Heart Rate**: Slider (71-202)
   - **Exercise Induced Angina**: Select box (Yes/No)
   - **ST Depression**: Slider (0.0-6.2)
   - **Slope of Peak Exercise ST**: Select box (1-3)
   - **Number of Major Vessels**: Select box (0-3)
   - **Thalassemia**: Select box (3.0, 6.0, 7.0)
4. Click "Predict" to generate predictions

### CSV Upload Mode

1. Navigate to the "Make Predictions" tab
2. Select "CSV Upload" input mode
3. Upload a CSV file with the following columns:
   ```
   age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
   ```
4. Click "Predict All Records" to process batch predictions
5. Download results as CSV

### Viewing Results

The app provides:

- **Individual model predictions**: Separate results for Logistic Regression and Random Forest
- **Disease probability**: Likelihood of heart disease (0-100%)
- **Confidence score**: Model confidence in the prediction
- **Ensemble recommendation**: Combined verdict from both models
- **Visual indicators**: Color-coded risk levels

### Exploring Interpretability

Navigate to the "Interpretability" tab to view:

- **SHAP Summary Plot**: Feature impact visualization
- **SHAP Bar Plot**: Aggregated feature importance
- **Feature Importance**: Model-specific feature rankings
- **Key Insights**: Clinical interpretation of results

### Checking Model Performance

Navigate to the "Model Performance" tab to see:

- Accuracy, Precision, Recall, F1, and ROC-AUC scores
- Comparison between models
- Metric definitions

### Understanding Fairness

Navigate to the "Fairness Analysis" tab to review:

- Performance breakdown by sex
- Performance breakdown by age group
- Fairness considerations and recommendations

## Input Data Schema

### Required Features

| Feature | Type | Range/Values | Description |
|---------|------|--------------|-------------|
| age | Numeric | 29-77 | Age in years |
| sex | Categorical | 0, 1 | 0: Female, 1: Male |
| cp | Categorical | 1-4 | Chest pain type |
| trestbps | Numeric | 94-200 | Resting blood pressure (mm Hg) |
| chol | Numeric | 126-564 | Serum cholesterol (mg/dl) |
| fbs | Categorical | 0, 1 | Fasting blood sugar > 120 mg/dl |
| restecg | Categorical | 0-2 | Resting ECG results |
| thalach | Numeric | 71-202 | Maximum heart rate achieved |
| exang | Categorical | 0, 1 | Exercise induced angina |
| oldpeak | Numeric | 0.0-6.2 | ST depression induced by exercise |
| slope | Categorical | 1-3 | Slope of peak exercise ST segment |
| ca | Categorical | 0-3 | Number of major vessels colored by fluoroscopy |
| thal | Categorical | 3.0, 6.0, 7.0 | Thalassemia type |

### CSV Format Example

```csv
age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal
63,1,1,145,233,1,2,150,0,2.3,3,0,6.0
50,0,2,130,204,0,0,172,0,1.4,2,0,3.0
```

## Regenerating Assets

If models or visualizations are missing, run the training pipeline:

```bash
cd Group07_XAI_Assignment_MayankSinghal_UtkarshShukla/XAI_HeartDisease
python scripts/run_pipeline.py
```

This will:
1. Download the UCI Heart Disease dataset
2. Train models (Logistic Regression and Random Forest)
3. Generate evaluation metrics
4. Create SHAP visualizations
5. Perform fairness analysis
6. Save all artifacts

## Project Structure

```
XAI_HeartDisease/
├── app/
│   ├── streamlit_app.py      # Main Streamlit application
│   └── README.md              # This file
├── pipeline/
│   └── core.py                # Core ML pipeline utilities
└── notebooks/                 # Jupyter notebooks

Group07_XAI_Assignment_MayankSinghal_UtkarshShukla/XAI_HeartDisease/
├── app/
│   └── models/                # Trained models and metadata
│       ├── logistic_regression_model.pkl
│       ├── random_forest_model.pkl
│       └── preprocessing_metadata.json
├── visuals/                   # Generated visualizations
│   ├── shap_summary.png
│   ├── shap_bar.png
│   ├── feature_importance.png
│   ├── metrics.csv
│   ├── fairness_by_sex.csv
│   └── fairness_by_age.csv
├── data/                      # Dataset
└── scripts/                   # Training scripts
```

## Troubleshooting

### Issue: Models not found

**Solution**: Ensure you've run the training pipeline or that the model files exist in:
```
Group07_XAI_Assignment_MayankSinghal_UtkarshShukla/XAI_HeartDisease/app/models/
```

### Issue: Visualizations not displaying

**Solution**: Check that PNG files exist in:
```
Group07_XAI_Assignment_MayankSinghal_UtkarshShukla/XAI_HeartDisease/visuals/
```

### Issue: Import errors

**Solution**: Install missing dependencies:
```bash
pip install streamlit pandas numpy scikit-learn joblib pillow
```

### Issue: Prediction errors

**Solution**: Verify input data:
- All 13 features are present
- Feature values are within valid ranges
- Categorical features use correct encoding

## Disclaimer

⚠️ **This application is for educational and research purposes only.**

It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified health providers with any questions regarding medical conditions.

## Technical Details

### Models

- **Logistic Regression**: Linear classifier with L2 regularization
  - Solver: liblinear
  - Class weight: balanced
  - Max iterations: 1000

- **Random Forest**: Ensemble of decision trees
  - Estimators: 200
  - Max depth: 8
  - Class weight: balanced

### Preprocessing

- **Numeric features**: Median imputation + StandardScaler
- **Categorical features**: Most frequent imputation + One-hot encoding
- **Output**: 28 features (5 numeric + 23 one-hot encoded categorical)

### Explainability

- **SHAP**: TreeExplainer for Random Forest, LinearExplainer for Logistic Regression
- **Feature importance**: Based on model coefficients or tree-based importance

## Support

For issues, questions, or contributions, please refer to the main project repository.

## License

See project root for license information.
