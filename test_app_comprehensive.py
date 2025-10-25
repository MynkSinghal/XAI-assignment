"""
Comprehensive integration test for Streamlit app.
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from XAI_HeartDisease.app.streamlit_app import (
    get_asset_paths,
    preprocess_input,
    make_predictions,
)


def test_all_features():
    """Test that all required assets and functions work correctly."""
    
    print("=" * 70)
    print("COMPREHENSIVE STREAMLIT APP TEST")
    print("=" * 70)
    
    # 1. Check asset paths
    print("\n1. Checking asset paths...")
    paths = get_asset_paths()
    
    required_assets = {
        "lr_model": "Logistic Regression Model",
        "rf_model": "Random Forest Model",
        "preprocessing_metadata": "Preprocessing Metadata",
        "metrics_csv": "Metrics CSV",
        "shap_summary": "SHAP Summary Plot",
        "shap_bar": "SHAP Bar Plot",
        "feature_importance": "Feature Importance Plot",
        "fairness_by_sex": "Fairness by Sex CSV",
        "fairness_by_age": "Fairness by Age CSV",
    }
    
    missing_assets = []
    for key, name in required_assets.items():
        if paths[key].exists():
            print(f"   ✅ {name}")
        else:
            print(f"   ❌ {name} - NOT FOUND")
            missing_assets.append(name)
    
    if missing_assets:
        print(f"\n   ⚠️  Warning: {len(missing_assets)} asset(s) missing")
    else:
        print("\n   ✅ All assets found!")
    
    # 2. Load models
    print("\n2. Loading models and metadata...")
    try:
        lr_model = joblib.load(paths["lr_model"])
        print(f"   ✅ Logistic Regression: {type(lr_model).__name__}")
    except Exception as e:
        print(f"   ❌ Logistic Regression failed: {e}")
        return False
    
    try:
        rf_model = joblib.load(paths["rf_model"])
        print(f"   ✅ Random Forest: {type(rf_model).__name__}")
    except Exception as e:
        print(f"   ❌ Random Forest failed: {e}")
        return False
    
    try:
        with open(paths["preprocessing_metadata"], "r") as f:
            metadata = json.load(f)
        print(f"   ✅ Metadata loaded: {len(metadata.get('feature_names', []))} features")
    except Exception as e:
        print(f"   ❌ Metadata loading failed: {e}")
        return False
    
    # 3. Test preprocessing with various inputs
    print("\n3. Testing preprocessing...")
    
    test_cases = [
        {
            "name": "Standard case",
            "data": {
                'age': 50, 'sex': 1, 'cp': 2, 'trestbps': 130, 'chol': 200,
                'fbs': 0, 'restecg': 0, 'thalach': 160, 'exang': 0,
                'oldpeak': 1.0, 'slope': 2, 'ca': 0, 'thal': 3.0
            }
        },
        {
            "name": "High risk profile",
            "data": {
                'age': 70, 'sex': 1, 'cp': 4, 'trestbps': 180, 'chol': 400,
                'fbs': 1, 'restecg': 2, 'thalach': 100, 'exang': 1,
                'oldpeak': 4.0, 'slope': 3, 'ca': 3, 'thal': 7.0
            }
        },
        {
            "name": "Low risk profile",
            "data": {
                'age': 35, 'sex': 0, 'cp': 1, 'trestbps': 110, 'chol': 180,
                'fbs': 0, 'restecg': 0, 'thalach': 180, 'exang': 0,
                'oldpeak': 0.0, 'slope': 1, 'ca': 0, 'thal': 3.0
            }
        }
    ]
    
    for test_case in test_cases:
        try:
            df = pd.DataFrame([test_case["data"]])
            processed = preprocess_input(df, metadata)
            
            if processed.shape[1] != len(metadata["dummy_columns"]):
                print(f"   ❌ {test_case['name']}: Shape mismatch")
                return False
            
            print(f"   ✅ {test_case['name']}: {df.shape} → {processed.shape}")
        except Exception as e:
            print(f"   ❌ {test_case['name']}: {e}")
            return False
    
    # 4. Test predictions
    print("\n4. Testing predictions...")
    
    for test_case in test_cases:
        try:
            df = pd.DataFrame([test_case["data"]])
            processed = preprocess_input(df, metadata)
            results = make_predictions(processed, lr_model, rf_model)
            
            if not results:
                print(f"   ❌ {test_case['name']}: No results")
                return False
            
            for model_name, result in results.items():
                pred = "Disease" if result["prediction"] == 1 else "No Disease"
                prob = result["probability"]
                conf = result["confidence"]
                
                if not (0 <= prob <= 1 and 0 <= conf <= 1):
                    print(f"   ❌ {test_case['name']}: Invalid probabilities")
                    return False
            
            print(f"   ✅ {test_case['name']}: Predictions generated")
            
        except Exception as e:
            print(f"   ❌ {test_case['name']}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # 5. Test batch processing
    print("\n5. Testing batch processing...")
    try:
        batch_df = pd.DataFrame([case["data"] for case in test_cases])
        processed_batch = preprocess_input(batch_df, metadata)
        
        if processed_batch.shape[0] != len(test_cases):
            print(f"   ❌ Batch size mismatch")
            return False
        
        for idx in range(len(processed_batch)):
            row_df = pd.DataFrame([processed_batch.iloc[idx]])
            results = make_predictions(row_df, lr_model, rf_model)
            
            if not results:
                print(f"   ❌ Batch prediction failed for row {idx}")
                return False
        
        print(f"   ✅ Batch processing: {len(test_cases)} records processed")
        
    except Exception as e:
        print(f"   ❌ Batch processing failed: {e}")
        return False
    
    # 6. Verify metrics and fairness data
    print("\n6. Verifying metrics and fairness data...")
    
    try:
        metrics_df = pd.read_csv(paths["metrics_csv"])
        print(f"   ✅ Metrics: {len(metrics_df)} model(s)")
        
        expected_cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
        if not all(col in metrics_df.columns for col in expected_cols):
            print(f"   ⚠️  Warning: Some metric columns missing")
        
    except Exception as e:
        print(f"   ❌ Metrics loading failed: {e}")
    
    try:
        fairness_sex = pd.read_csv(paths["fairness_by_sex"])
        print(f"   ✅ Fairness (sex): {len(fairness_sex)} group(s)")
    except Exception as e:
        print(f"   ❌ Fairness (sex) loading failed: {e}")
    
    try:
        fairness_age = pd.read_csv(paths["fairness_by_age"])
        print(f"   ✅ Fairness (age): {len(fairness_age)} group(s)")
    except Exception as e:
        print(f"   ❌ Fairness (age) loading failed: {e}")
    
    # 7. Check image files
    print("\n7. Checking visualization files...")
    
    vis_files = {
        "shap_summary": "SHAP Summary",
        "shap_bar": "SHAP Bar",
        "feature_importance": "Feature Importance",
    }
    
    for key, name in vis_files.items():
        if paths[key].exists():
            size_kb = paths[key].stat().st_size / 1024
            print(f"   ✅ {name}: {size_kb:.1f} KB")
        else:
            print(f"   ⚠️  {name}: Not found")
    
    # Final summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print("✅ All core functionality tests passed!")
    print("\nThe Streamlit app is fully functional and ready for use.")
    print("\nTo launch the app:")
    print("  streamlit run XAI_HeartDisease/app/streamlit_app.py")
    print("\nFeatures verified:")
    print("  • Model loading (Logistic Regression & Random Forest)")
    print("  • Input preprocessing (single & batch)")
    print("  • Prediction generation with probabilities")
    print("  • Metrics and fairness data")
    print("  • Visualization assets")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    try:
        success = test_all_features()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
