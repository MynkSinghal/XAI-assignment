# Streamlit App Quick Start Guide

## Overview

The Explainable AI Heart Disease Prediction Streamlit app is now available at:
```
XAI_HeartDisease/app/streamlit_app.py
```

## Quick Start

### 1. Launch the App

From the project root directory:

```bash
streamlit run XAI_HeartDisease/app/streamlit_app.py
```

The app will automatically open in your browser at `http://localhost:8501`

### 2. Using the App

The app has 5 main tabs:

#### 🔮 Make Predictions
- **Manual Entry**: Fill in a form with 13 clinical features
- **CSV Upload**: Upload a CSV file for batch predictions

#### 📊 Model Performance
- View accuracy, precision, recall, F1, and ROC-AUC scores
- Compare Logistic Regression vs Random Forest

#### 🔍 Interpretability
- Explore SHAP visualizations
- View feature importance rankings
- Understand key clinical insights

#### ⚖️ Fairness Analysis
- Review model performance across demographics
- Understand potential biases

#### ℹ️ About
- Learn about the technology and methodology
- Read disclaimers and ethical considerations

## Testing the App

Run the comprehensive test suite:

```bash
python test_app_comprehensive.py
```

This verifies:
- ✅ All models and assets are present
- ✅ Preprocessing works correctly
- ✅ Predictions generate successfully
- ✅ Batch processing functions properly
- ✅ Metrics and fairness data load correctly

## Features Implemented

### ✅ Dark Theme Styling
- Custom CSS for dark mode
- Responsive design for desktop browsers
- Color-coded predictions and alerts

### ✅ Dual Input Modes
- **Manual Entry Form**:
  - Sliders for numeric features
  - Select boxes for categorical features
  - Helpful tooltips with feature descriptions
  - Real-time validation
  
- **CSV Upload**:
  - Batch processing capability
  - Progress tracking
  - Downloadable results

### ✅ Model Integration
- Loads serialized Logistic Regression and Random Forest models
- Preprocessing aligns with training pipeline
- Consistent feature order and encoding

### ✅ Prediction Outputs
- Binary predictions (Heart Disease: Yes/No)
- Class probabilities (0-100%)
- Confidence scores
- Ensemble recommendations combining both models

### ✅ Evaluation Metrics
- Displays metrics from stored CSV
- Formatted tables with 4 decimal precision
- Metric definitions and explanations

### ✅ Interpretability Section
- SHAP summary plots
- SHAP bar plots (feature importance)
- Feature importance from best model
- Clinical insights and key findings

### ✅ Fairness Considerations
- Performance by sex (Male/Female)
- Performance by age group (<50, 50-60, 60+)
- Contextual explanations
- Ethical considerations

### ✅ Error Handling
- Graceful handling of missing models
- Clear error messages
- Instructions for regenerating assets
- Validation of input data

## Input Data Format

### Required Features (13 total)

| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| age | Age in years | Numeric | 29-77 |
| sex | Sex (0=Female, 1=Male) | Categorical | 0, 1 |
| cp | Chest pain type | Categorical | 1-4 |
| trestbps | Resting blood pressure (mm Hg) | Numeric | 94-200 |
| chol | Serum cholesterol (mg/dl) | Numeric | 126-564 |
| fbs | Fasting blood sugar > 120 mg/dl | Categorical | 0, 1 |
| restecg | Resting ECG results | Categorical | 0-2 |
| thalach | Maximum heart rate achieved | Numeric | 71-202 |
| exang | Exercise induced angina | Categorical | 0, 1 |
| oldpeak | ST depression by exercise | Numeric | 0.0-6.2 |
| slope | Slope of peak exercise ST | Categorical | 1-3 |
| ca | Number of major vessels | Categorical | 0-3 |
| thal | Thalassemia | Categorical | 3.0, 6.0, 7.0 |

### CSV Example

```csv
age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal
63,1,1,145,233,1,2,150,0,2.3,3,0,6.0
50,0,2,130,204,0,0,172,0,1.4,2,0,3.0
70,1,4,180,400,1,2,100,1,4.0,3,3,7.0
```

## Troubleshooting

### Issue: Models not loading

**Check**: Ensure models exist at:
```
Group07_XAI_Assignment_MayankSinghal_UtkarshShukla/XAI_HeartDisease/app/models/
```

**Solution**: Run the training pipeline to regenerate:
```bash
cd Group07_XAI_Assignment_MayankSinghal_UtkarshShukla/XAI_HeartDisease
python scripts/run_pipeline.py
```

### Issue: Visualizations not displaying

**Check**: PNG files should exist in:
```
Group07_XAI_Assignment_MayankSinghal_UtkarshShukla/XAI_HeartDisease/visuals/
```

**Solution**: Same as above - run the training pipeline.

### Issue: Import errors

**Solution**: Install dependencies:
```bash
pip install streamlit pandas numpy scikit-learn joblib pillow
```

## Architecture

### File Structure
```
XAI_HeartDisease/
├── app/
│   ├── streamlit_app.py       # Main application
│   └── README.md              # Detailed documentation
├── pipeline/
│   └── core.py                # Shared utilities
└── notebooks/                 # Analysis notebooks
```

### Data Flow
1. **Input** → Manual form or CSV upload
2. **Preprocessing** → One-hot encoding + feature alignment
3. **Prediction** → Both models generate predictions
4. **Output** → Probabilities + ensemble recommendation

### Models
- **Logistic Regression**: Linear baseline model
- **Random Forest**: Non-linear ensemble model

Both models:
- Use same preprocessing pipeline
- Trained with balanced class weights
- Optimized for ROC-AUC score

## Technical Details

### Preprocessing
- Numeric features: 5 (age, trestbps, chol, thalach, oldpeak)
- Categorical features: 8 (sex, cp, fbs, restecg, exang, slope, ca, thal)
- Output features: 28 (after one-hot encoding)

### Theme Configuration
The app uses custom CSS for dark theme with:
- Dark background (#0e1117)
- Light text (#fafafa)
- Color-coded alerts (success, warning, error)
- Gradient prediction boxes
- Responsive layout

## Acceptance Criteria Status

✅ **App launches without errors**: Verified with `streamlit run`  
✅ **Both input modes function**: Manual form and CSV upload tested  
✅ **Predictions consistent with models**: Verified with test suite  
✅ **SHAP and feature importance render**: PNG files display correctly  
✅ **Dark theme styling**: Custom CSS applied  
✅ **Responsive on desktop**: Layout adapts to browser width  

## Next Steps

1. **Run the app**:
   ```bash
   streamlit run XAI_HeartDisease/app/streamlit_app.py
   ```

2. **Test with sample data**: Use the manual form to try different patient profiles

3. **Explore interpretability**: Navigate through all tabs to see SHAP analysis

4. **Review fairness**: Check demographic performance breakdowns

## Support

For detailed documentation, see:
- `XAI_HeartDisease/app/README.md` - Full application documentation
- `test_app_comprehensive.py` - Comprehensive test suite
- `test_streamlit_app.py` - Basic validation tests

## Disclaimer

⚠️ **This application is for educational and research purposes only.**
It should not be used as a substitute for professional medical advice.
Always consult qualified healthcare providers for medical decisions.

---

**Status**: ✅ Ready for Production Use  
**Last Updated**: 2024  
**Version**: 1.0.0
