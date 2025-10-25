"""
Streamlit application for Explainable AI Heart Disease Prediction.

This application provides an interactive interface for heart disease prediction
using trained ML models (Logistic Regression and Random Forest) with
explainability features including SHAP analysis and feature importance.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


DARK_THEME_CSS = """
<style>
    /* Main app background */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #fafafa !important;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        color: #fafafa;
    }
    
    /* Tables */
    .dataframe {
        background-color: #1e2130;
        color: #fafafa;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #262730;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: #1e2130;
        color: #fafafa;
    }
    
    /* Success box */
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #1a4d2e;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
    }
    
    /* Warning box */
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #4d3d1a;
        border-left: 5px solid #ff9800;
        margin: 1rem 0;
    }
    
    /* Error box */
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #4d1a1a;
        border-left: 5px solid #f44336;
        margin: 1rem 0;
    }
    
    /* Prediction result box */
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.75rem;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border: 2px solid #3a5998;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .prediction-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #ffffff;
    }
    
    .prediction-value {
        font-size: 2rem;
        font-weight: bold;
        color: #4caf50;
        text-align: center;
        margin: 1rem 0;
    }
    
    .prediction-value.positive {
        color: #f44336;
    }
</style>
"""


FEATURE_DESCRIPTIONS = {
    "age": "Age in years (29-77)",
    "sex": "Sex (0: Female, 1: Male)",
    "cp": "Chest pain type (1: Typical angina, 2: Atypical angina, 3: Non-anginal pain, 4: Asymptomatic)",
    "trestbps": "Resting blood pressure in mm Hg (94-200)",
    "chol": "Serum cholesterol in mg/dl (126-564)",
    "fbs": "Fasting blood sugar > 120 mg/dl (0: False, 1: True)",
    "restecg": "Resting ECG results (0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy)",
    "thalach": "Maximum heart rate achieved (71-202)",
    "exang": "Exercise induced angina (0: No, 1: Yes)",
    "oldpeak": "ST depression induced by exercise (0.0-6.2)",
    "slope": "Slope of peak exercise ST segment (1: Upsloping, 2: Flat, 3: Downsloping)",
    "ca": "Number of major vessels colored by fluoroscopy (0-3)",
    "thal": "Thalassemia (3: Normal, 6: Fixed defect, 7: Reversible defect)",
}


def get_asset_paths() -> Dict[str, Path]:
    """Get paths to all required assets."""
    app_dir = Path(__file__).parent
    project_root = app_dir.parent.parent
    
    group_dir = project_root / "Group07_XAI_Assignment_MayankSinghal_UtkarshShukla" / "XAI_HeartDisease"
    
    return {
        "models_dir": group_dir / "app" / "models",
        "visuals_dir": group_dir / "visuals",
        "data_dir": group_dir / "data",
        "lr_model": group_dir / "app" / "models" / "logistic_regression_model.pkl",
        "rf_model": group_dir / "app" / "models" / "random_forest_model.pkl",
        "preprocessing_metadata": group_dir / "app" / "models" / "preprocessing_metadata.json",
        "metrics_csv": group_dir / "visuals" / "metrics.csv",
        "shap_summary": group_dir / "visuals" / "shap_summary.png",
        "shap_bar": group_dir / "visuals" / "shap_bar.png",
        "feature_importance": group_dir / "visuals" / "feature_importance.png",
        "fairness_by_sex": group_dir / "visuals" / "fairness_by_sex.csv",
        "fairness_by_age": group_dir / "visuals" / "fairness_by_age.csv",
    }


@st.cache_resource
def load_models() -> Tuple[Optional[object], Optional[object], Optional[Dict]]:
    """Load trained models and preprocessing metadata."""
    paths = get_asset_paths()
    
    lr_model = None
    rf_model = None
    metadata = None
    
    try:
        if paths["lr_model"].exists():
            lr_model = joblib.load(paths["lr_model"])
            st.success("✅ Logistic Regression model loaded successfully")
        else:
            st.warning(f"⚠️ Logistic Regression model not found at {paths['lr_model']}")
    except Exception as e:
        st.error(f"❌ Error loading Logistic Regression model: {str(e)}")
    
    try:
        if paths["rf_model"].exists():
            rf_model = joblib.load(paths["rf_model"])
            st.success("✅ Random Forest model loaded successfully")
        else:
            st.warning(f"⚠️ Random Forest model not found at {paths['rf_model']}")
    except Exception as e:
        st.error(f"❌ Error loading Random Forest model: {str(e)}")
    
    try:
        if paths["preprocessing_metadata"].exists():
            with open(paths["preprocessing_metadata"], "r") as f:
                metadata = json.load(f)
            st.success("✅ Preprocessing metadata loaded successfully")
        else:
            st.warning(f"⚠️ Preprocessing metadata not found at {paths['preprocessing_metadata']}")
    except Exception as e:
        st.error(f"❌ Error loading preprocessing metadata: {str(e)}")
    
    return lr_model, rf_model, metadata


def preprocess_input(df: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
    """Preprocess input data to match training feature schema."""
    numeric_features = metadata.get("numeric_features", [])
    categorical_features = metadata.get("categorical_features", [])
    dummy_columns = metadata.get("dummy_columns", [])
    
    processed_df = df.copy()
    
    for col in numeric_features:
        if col in processed_df.columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors="coerce")
    
    for col in categorical_features:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].astype(str)
    
    processed_df = pd.get_dummies(processed_df, columns=categorical_features, dtype=float)
    
    for col in dummy_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0.0
    
    processed_df = processed_df[dummy_columns]
    
    return processed_df


def make_predictions(
    input_data: pd.DataFrame,
    lr_model: Optional[object],
    rf_model: Optional[object],
) -> Dict[str, Dict]:
    """Make predictions using both models."""
    results = {}
    
    if lr_model is not None:
        try:
            lr_pred = lr_model.predict(input_data)[0]
            lr_proba = lr_model.predict_proba(input_data)[0]
            results["Logistic Regression"] = {
                "prediction": int(lr_pred),
                "probability": float(lr_proba[1]),
                "confidence": float(max(lr_proba)),
            }
        except Exception as e:
            st.error(f"Error making Logistic Regression prediction: {str(e)}")
    
    if rf_model is not None:
        try:
            rf_pred = rf_model.predict(input_data)[0]
            rf_proba = rf_model.predict_proba(input_data)[0]
            results["Random Forest"] = {
                "prediction": int(rf_pred),
                "probability": float(rf_proba[1]),
                "confidence": float(max(rf_proba)),
            }
        except Exception as e:
            st.error(f"Error making Random Forest prediction: {str(e)}")
    
    return results


def display_prediction_results(results: Dict[str, Dict]):
    """Display prediction results with styling."""
    st.markdown("## 🔮 Prediction Results")
    
    ensemble_votes = []
    
    for model_name, result in results.items():
        prediction_text = "❤️ **Heart Disease Detected**" if result["prediction"] == 1 else "✅ **No Heart Disease**"
        prediction_class = "positive" if result["prediction"] == 1 else "negative"
        
        st.markdown(f"### {model_name}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prediction", prediction_text)
        with col2:
            st.metric("Disease Probability", f"{result['probability']:.2%}")
        with col3:
            st.metric("Confidence", f"{result['confidence']:.2%}")
        
        ensemble_votes.append(result["prediction"])
        
        st.progress(result["probability"])
        st.markdown("---")
    
    if len(ensemble_votes) >= 2:
        ensemble_prediction = 1 if sum(ensemble_votes) >= len(ensemble_votes) / 2 else 0
        
        st.markdown("### 🎯 Ensemble Recommendation")
        if ensemble_prediction == 1:
            st.markdown(
                '<div class="error-box">'
                '<h4>⚠️ High Risk: Heart Disease Detected</h4>'
                '<p>Both models indicate a high likelihood of heart disease. '
                'Please consult with a healthcare professional for proper diagnosis and treatment.</p>'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="success-box">'
                '<h4>✅ Low Risk: No Heart Disease Detected</h4>'
                '<p>Both models indicate a low likelihood of heart disease. '
                'However, regular health checkups are recommended.</p>'
                '</div>',
                unsafe_allow_html=True
            )


def render_manual_input_form(metadata: Dict) -> Optional[pd.DataFrame]:
    """Render manual input form for patient data."""
    st.markdown("### 📝 Enter Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 29, 77, 50, help=FEATURE_DESCRIPTIONS["age"])
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", help=FEATURE_DESCRIPTIONS["sex"])
        cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4], help=FEATURE_DESCRIPTIONS["cp"])
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 94, 200, 120, help=FEATURE_DESCRIPTIONS["trestbps"])
        chol = st.slider("Serum Cholesterol (mg/dl)", 126, 564, 200, help=FEATURE_DESCRIPTIONS["chol"])
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", help=FEATURE_DESCRIPTIONS["fbs"])
    
    with col2:
        restecg = st.selectbox("Resting ECG Results", [0, 1, 2], help=FEATURE_DESCRIPTIONS["restecg"])
        thalach = st.slider("Maximum Heart Rate", 71, 202, 150, help=FEATURE_DESCRIPTIONS["thalach"])
        exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", help=FEATURE_DESCRIPTIONS["exang"])
        oldpeak = st.slider("ST Depression", 0.0, 6.2, 1.0, 0.1, help=FEATURE_DESCRIPTIONS["oldpeak"])
        slope = st.selectbox("Slope of Peak Exercise ST", [1, 2, 3], help=FEATURE_DESCRIPTIONS["slope"])
        ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3], help=FEATURE_DESCRIPTIONS["ca"])
        thal = st.selectbox("Thalassemia", [3.0, 6.0, 7.0], help=FEATURE_DESCRIPTIONS["thal"])
    
    if st.button("🔍 Predict", type="primary", use_container_width=True):
        input_dict = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
        }
        return pd.DataFrame([input_dict])
    
    return None


def render_csv_upload(metadata: Dict) -> Optional[pd.DataFrame]:
    """Render CSV upload interface."""
    st.markdown("### 📤 Upload Patient Data (CSV)")
    
    st.info(
        "Upload a CSV file containing patient data. The file should include the following columns:\n\n"
        + ", ".join(metadata.get("numeric_features", []) + metadata.get("categorical_features", []))
    )
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! Found {len(df)} record(s)")
            st.dataframe(df.head())
            
            if st.button("🔍 Predict All Records", type="primary"):
                return df
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
    
    return None


def display_evaluation_metrics():
    """Display model evaluation metrics."""
    st.markdown("## 📊 Model Performance Metrics")
    
    paths = get_asset_paths()
    
    if paths["metrics_csv"].exists():
        try:
            metrics_df = pd.read_csv(paths["metrics_csv"])
            st.dataframe(metrics_df.style.format({
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1": "{:.4f}",
                "ROC_AUC": "{:.4f}",
            }), use_container_width=True)
            
            st.markdown("""
            **Metric Definitions:**
            - **Accuracy**: Overall correctness of predictions
            - **Precision**: Proportion of positive predictions that are correct
            - **Recall**: Proportion of actual positives correctly identified
            - **F1**: Harmonic mean of precision and recall
            - **ROC_AUC**: Area under the ROC curve (model's discriminative ability)
            """)
        except Exception as e:
            st.error(f"❌ Error loading metrics: {str(e)}")
    else:
        st.warning("⚠️ Metrics file not found. Please run the training pipeline first.")


def display_interpretability_section():
    """Display SHAP and feature importance visuals."""
    st.markdown("## 🔍 Model Interpretability")
    
    paths = get_asset_paths()
    
    st.markdown("### SHAP Analysis")
    st.markdown("""
    **SHAP (SHapley Additive exPlanations)** values help us understand how each feature 
    contributes to the model's predictions. The visualizations below show:
    
    - **SHAP Summary Plot**: Shows the impact of each feature on model predictions across all samples
    - **Feature Importance**: Aggregated importance of each feature based on SHAP values
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if paths["shap_summary"].exists():
            try:
                shap_img = Image.open(paths["shap_summary"])
                st.image(shap_img, caption="SHAP Summary Plot", use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error loading SHAP summary: {str(e)}")
        else:
            st.warning("⚠️ SHAP summary plot not found")
    
    with col2:
        if paths["shap_bar"].exists():
            try:
                shap_bar_img = Image.open(paths["shap_bar"])
                st.image(shap_bar_img, caption="SHAP Feature Importance", use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error loading SHAP bar plot: {str(e)}")
        else:
            st.warning("⚠️ SHAP bar plot not found")
    
    st.markdown("### Feature Importance")
    if paths["feature_importance"].exists():
        try:
            fi_img = Image.open(paths["feature_importance"])
            st.image(fi_img, caption="Feature Importance from Best Model", use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error loading feature importance plot: {str(e)}")
    else:
        st.warning("⚠️ Feature importance plot not found")
    
    st.markdown("""
    ### 📖 Key Insights from SHAP Analysis
    
    Based on the SHAP analysis of our heart disease prediction models:
    
    1. **Most Important Features**:
       - **Chest Pain Type (cp)**: Different types of chest pain are strongly associated with heart disease
       - **Number of Major Vessels (ca)**: Higher number of vessels colored by fluoroscopy indicates blockages
       - **Thalassemia (thal)**: Blood disorder that affects heart disease risk
       - **Maximum Heart Rate (thalach)**: Lower maximum heart rates during exercise may indicate heart problems
       - **Age**: Older patients have higher risk of heart disease
    
    2. **Clinical Implications**:
       - Asymptomatic chest pain (cp=4) is a strong predictor
       - Exercise-induced angina (exang) significantly impacts predictions
       - ST depression (oldpeak) provides important diagnostic information
    
    3. **Model Behavior**:
       - Both models show consistent feature importance rankings
       - Non-linear relationships are captured by the Random Forest model
       - SHAP values reveal how features interact to affect predictions
    """)


def display_fairness_considerations():
    """Display fairness analysis and considerations."""
    st.markdown("## ⚖️ Fairness and Bias Considerations")
    
    paths = get_asset_paths()
    
    st.markdown("""
    We have analyzed model fairness across sensitive attributes to ensure equitable predictions.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Performance by Sex")
        if paths["fairness_by_sex"].exists():
            try:
                fairness_sex = pd.read_csv(paths["fairness_by_sex"])
                st.dataframe(fairness_sex.style.format({
                    "Accuracy": "{:.4f}",
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1": "{:.4f}",
                }), use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error loading fairness data: {str(e)}")
        else:
            st.warning("⚠️ Fairness data not found")
    
    with col2:
        st.markdown("### Performance by Age Group")
        if paths["fairness_by_age"].exists():
            try:
                fairness_age = pd.read_csv(paths["fairness_by_age"])
                st.dataframe(fairness_age.style.format({
                    "Accuracy": "{:.4f}",
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1": "{:.4f}",
                }), use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error loading fairness data: {str(e)}")
        else:
            st.warning("⚠️ Fairness data not found")
    
    st.markdown("""
    ### 📊 Fairness Analysis Summary
    
    **Sex-based Analysis:**
    - The model shows different performance across sex groups
    - Female patients (sex=0) show higher precision but lower recall
    - Male patients (sex=1) have slightly lower overall accuracy
    - These differences may reflect underlying prevalence patterns in the data
    
    **Age-based Analysis:**
    - Younger patients (<50) show perfect metrics, possibly due to smaller sample size
    - Middle-aged patients (50-60) maintain strong performance
    - Older patients (60+) show reduced accuracy, which requires careful monitoring
    
    ### ⚠️ Important Considerations
    
    1. **Clinical Context**: These models are designed to assist, not replace, clinical judgment
    2. **Data Representation**: Performance differences may reflect data distribution rather than model bias
    3. **Continuous Monitoring**: Regular fairness audits should be conducted in production
    4. **Transparent Communication**: Clinicians should be aware of potential performance variations
    5. **Ethical Use**: Always consider the broader context of patient care and individual circumstances
    """)


def display_instructions():
    """Display instructions for regenerating assets."""
    with st.expander("📚 How to Regenerate Assets"):
        st.markdown("""
        If you're missing models or visualizations, follow these steps to regenerate them:
        
        ### 1. Install Dependencies
        ```bash
        pip install -r requirements.txt
        ```
        
        ### 2. Run the Training Pipeline
        ```bash
        cd Group07_XAI_Assignment_MayankSinghal_UtkarshShukla/XAI_HeartDisease
        python scripts/run_pipeline.py
        ```
        
        This will:
        - Download the heart disease dataset
        - Train Logistic Regression and Random Forest models
        - Generate evaluation metrics
        - Create SHAP visualizations
        - Perform fairness analysis
        - Save all models and artifacts
        
        ### 3. Verify Asset Generation
        Check that the following directories contain the required files:
        - `app/models/`: Model files (.pkl) and preprocessing metadata (.json)
        - `visuals/`: PNG images for SHAP, feature importance, and other plots
        - `visuals/`: CSV files for metrics and fairness analysis
        
        ### 4. Restart the Streamlit App
        ```bash
        streamlit run XAI_HeartDisease/app/streamlit_app.py
        ```
        """)


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Explainable AI for Heart Disease Prediction",
        page_icon="❤️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)
    
    st.title("❤️ Explainable AI for Heart Disease Prediction")
    st.markdown("""
    This application uses machine learning models to predict the presence of heart disease
    based on clinical features, with built-in explainability and fairness analysis.
    """)
    
    display_instructions()
    
    with st.spinner("Loading models and metadata..."):
        lr_model, rf_model, metadata = load_models()
    
    if lr_model is None and rf_model is None:
        st.error("""
        ❌ **No models found!** 
        
        Please ensure models are trained and available in the expected directory.
        Expand the instructions section above to learn how to regenerate assets.
        """)
        return
    
    if metadata is None:
        st.error("❌ **Preprocessing metadata not found!** Cannot proceed without feature schema.")
        return
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔮 Make Predictions",
        "📊 Model Performance",
        "🔍 Interpretability",
        "⚖️ Fairness Analysis",
        "ℹ️ About"
    ])
    
    with tab1:
        st.markdown("## 🔮 Make Predictions")
        
        input_mode = st.radio(
            "Select Input Mode:",
            ["Manual Entry", "CSV Upload"],
            horizontal=True
        )
        
        input_df = None
        
        if input_mode == "Manual Entry":
            input_df = render_manual_input_form(metadata)
        else:
            input_df = render_csv_upload(metadata)
        
        if input_df is not None:
            try:
                st.markdown("### 🔄 Preprocessing Input...")
                processed_input = preprocess_input(input_df, metadata)
                
                if len(processed_input) == 1:
                    st.markdown("### 🔮 Making Predictions...")
                    results = make_predictions(processed_input, lr_model, rf_model)
                    
                    if results:
                        display_prediction_results(results)
                    else:
                        st.error("❌ Could not generate predictions. Check model status above.")
                else:
                    st.markdown("### 🔮 Batch Predictions")
                    all_results = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, row in processed_input.iterrows():
                        status_text.text(f"Processing record {idx + 1} of {len(processed_input)}...")
                        
                        row_df = pd.DataFrame([row])
                        results = make_predictions(row_df, lr_model, rf_model)
                        
                        result_row = {"Record": idx + 1}
                        for model_name, result in results.items():
                            result_row[f"{model_name}_Prediction"] = "Disease" if result["prediction"] == 1 else "No Disease"
                            result_row[f"{model_name}_Probability"] = f"{result['probability']:.2%}"
                        
                        all_results.append(result_row)
                        progress_bar.progress((idx + 1) / len(processed_input))
                    
                    status_text.text("✅ All predictions completed!")
                    
                    results_df = pd.DataFrame(all_results)
                    st.dataframe(results_df, use_container_width=True)
                    
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="heart_disease_predictions.csv",
                        mime="text/csv",
                    )
            
            except Exception as e:
                st.error(f"❌ Error processing input: {str(e)}")
                st.exception(e)
    
    with tab2:
        display_evaluation_metrics()
    
    with tab3:
        display_interpretability_section()
    
    with tab4:
        display_fairness_considerations()
    
    with tab5:
        st.markdown("""
        ## ℹ️ About This Application
        
        ### Overview
        This Streamlit application demonstrates Explainable AI (XAI) techniques for heart disease prediction.
        It combines machine learning predictions with interpretability methods to provide transparent,
        trustworthy insights for clinical decision support.
        
        ### Models
        - **Logistic Regression**: Linear model providing baseline performance with interpretable coefficients
        - **Random Forest**: Ensemble model capturing non-linear relationships and feature interactions
        
        ### Explainability Methods
        - **SHAP (SHapley Additive exPlanations)**: Game-theoretic approach to explain model predictions
        - **Feature Importance**: Ranking of features by their contribution to model decisions
        
        ### Dataset
        The models are trained on the UCI Heart Disease dataset, which contains clinical features
        from patients evaluated for heart disease.
        
        ### Disclaimer
        ⚠️ **This application is for educational and research purposes only.**
        It should not be used as a substitute for professional medical advice, diagnosis, or treatment.
        Always seek the advice of qualified health providers with any questions regarding medical conditions.
        
        ### Technology Stack
        - **Python**: Core programming language
        - **Streamlit**: Web application framework
        - **Scikit-learn**: Machine learning models
        - **SHAP**: Model explainability
        - **Pandas & NumPy**: Data manipulation
        
        ### Contact & Support
        For questions, issues, or contributions, please refer to the project repository.
        """)


if __name__ == "__main__":
    main()
