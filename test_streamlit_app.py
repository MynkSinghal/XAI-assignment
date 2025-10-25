"""
Quick test script to verify Streamlit app functionality.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        from XAI_HeartDisease.app.streamlit_app import (
            get_asset_paths,
            load_models,
            preprocess_input,
            make_predictions,
        )
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_asset_paths():
    """Test that asset paths are correctly configured."""
    print("\nTesting asset paths...")
    from XAI_HeartDisease.app.streamlit_app import get_asset_paths
    
    paths = get_asset_paths()
    required_assets = [
        "lr_model",
        "rf_model",
        "preprocessing_metadata",
        "metrics_csv",
        "shap_summary",
        "feature_importance",
    ]
    
    all_found = True
    for asset in required_assets:
        if paths[asset].exists():
            print(f"✅ {asset}: Found")
        else:
            print(f"❌ {asset}: Not found at {paths[asset]}")
            all_found = False
    
    return all_found


def test_preprocessing_and_prediction():
    """Test preprocessing and prediction pipeline."""
    print("\nTesting preprocessing and prediction...")
    
    import json
    import joblib
    import pandas as pd
    from pathlib import Path
    from XAI_HeartDisease.app.streamlit_app import (
        get_asset_paths,
        preprocess_input,
        make_predictions,
    )
    
    try:
        paths = get_asset_paths()
        
        # Load models and metadata
        lr_model = joblib.load(paths["lr_model"])
        rf_model = joblib.load(paths["rf_model"])
        
        with open(paths["preprocessing_metadata"], "r") as f:
            metadata = json.load(f)
        
        # Create sample patient data
        sample_data = pd.DataFrame([{
            'age': 50,
            'sex': 1,
            'cp': 2,
            'trestbps': 130,
            'chol': 200,
            'fbs': 0,
            'restecg': 0,
            'thalach': 160,
            'exang': 0,
            'oldpeak': 1.0,
            'slope': 2,
            'ca': 0,
            'thal': 3.0
        }])
        
        # Preprocess
        processed = preprocess_input(sample_data, metadata)
        print(f"  Input shape: {sample_data.shape}")
        print(f"  Processed shape: {processed.shape}")
        
        # Make predictions
        results = make_predictions(processed, lr_model, rf_model)
        
        print("  Predictions:")
        for model_name, result in results.items():
            pred_text = "Disease" if result["prediction"] == 1 else "No Disease"
            print(f"    {model_name}: {pred_text} (prob: {result['probability']:.2%})")
        
        print("✅ Preprocessing and prediction successful")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Streamlit App Validation Tests")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_asset_paths,
        test_preprocessing_and_prediction,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    
    if all(results):
        print("\n🎉 All tests passed! The Streamlit app is ready to use.")
        print("\nTo run the app:")
        print("  streamlit run XAI_HeartDisease/app/streamlit_app.py")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
