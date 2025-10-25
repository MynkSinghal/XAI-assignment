#!/usr/bin/env python3
"""
Validation script to verify all pipeline outputs meet acceptance criteria.
"""

import json
from pathlib import Path
import joblib
import pandas as pd


def main():
    """Validate pipeline outputs."""
    project_root = Path(__file__).parent.parent
    
    print("=" * 70)
    print("VALIDATING PIPELINE OUTPUTS")
    print("=" * 70)
    
    checks_passed = 0
    checks_total = 0
    
    # Check 1: Data file exists
    checks_total += 1
    data_file = project_root / "data" / "heart_disease.csv"
    if data_file.exists():
        df = pd.read_csv(data_file)
        print(f"✓ Processed dataset exists: {df.shape[0]} rows, {df.shape[1]} columns")
        checks_passed += 1
    else:
        print("✗ Processed dataset not found")
    
    # Check 2: Models exist and loadable
    checks_total += 1
    lr_model_path = project_root / "app" / "models" / "logistic_regression_model.pkl"
    rf_model_path = project_root / "app" / "models" / "random_forest_model.pkl"
    if lr_model_path.exists() and rf_model_path.exists():
        try:
            lr_model = joblib.load(lr_model_path)
            rf_model = joblib.load(rf_model_path)
            print(f"✓ Models loaded successfully (LR, RF)")
            checks_passed += 1
        except Exception as e:
            print(f"✗ Failed to load models: {e}")
    else:
        print("✗ Model files not found")
    
    # Check 3: Preprocessing metadata exists
    checks_total += 1
    metadata_path = project_root / "app" / "models" / "preprocessing_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"✓ Preprocessing metadata exists ({len(metadata['feature_names'])} features)")
        checks_passed += 1
    else:
        print("✗ Preprocessing metadata not found")
    
    # Check 4: Metrics CSV with both models
    checks_total += 1
    metrics_path = project_root / "visuals" / "metrics.csv"
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        required_cols = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC']
        if all(col in metrics_df.columns for col in required_cols):
            print(f"✓ Metrics table exists with {len(metrics_df)} models and all required scores")
            checks_passed += 1
        else:
            print(f"✗ Metrics table missing required columns")
    else:
        print("✗ Metrics CSV not found")
    
    # Check 5: Visualizations exist
    checks_total += 1
    required_visuals = [
        "confusion_matrix.png",
        "roc_curves.png",
        "feature_importance.png",
        "shap_summary.png",
        "shap_bar.png"
    ]
    visuals_dir = project_root / "visuals"
    existing_visuals = [v for v in required_visuals if (visuals_dir / v).exists()]
    if len(existing_visuals) == len(required_visuals):
        print(f"✓ All required visualizations exist ({len(existing_visuals)}/{len(required_visuals)})")
        checks_passed += 1
    else:
        print(f"✗ Missing visualizations: {set(required_visuals) - set(existing_visuals)}")
    
    # Check 6: Fairness CSVs exist
    checks_total += 1
    fairness_sex_path = visuals_dir / "fairness_by_sex.csv"
    fairness_age_path = visuals_dir / "fairness_by_age.csv"
    if fairness_sex_path.exists() and fairness_age_path.exists():
        df_sex = pd.read_csv(fairness_sex_path)
        df_age = pd.read_csv(fairness_age_path)
        print(f"✓ Fairness analysis exists (sex: {len(df_sex)} groups, age: {len(df_age)} groups)")
        checks_passed += 1
    else:
        print("✗ Fairness CSV files not found")
    
    # Check 7: Analytical narrative exists
    checks_total += 1
    narrative_path = project_root / "report" / "analytical_narrative.md"
    if narrative_path.exists():
        with open(narrative_path, 'r') as f:
            narrative = f.read()
        word_count = len(narrative.split())
        if 500 <= word_count <= 1000:
            print(f"✓ Analytical narrative exists ({word_count} words, target: 500-700)")
            checks_passed += 1
        else:
            print(f"⚠ Analytical narrative exists but word count is {word_count} (target: 500-700)")
            checks_passed += 0.5
    else:
        print("✗ Analytical narrative not found")
    
    # Check 8: README exists
    checks_total += 1
    readme_path = project_root / "README.md"
    if readme_path.exists():
        print(f"✓ README exists")
        checks_passed += 1
    else:
        print("✗ README not found")
    
    # Check 9: requirements.txt exists
    checks_total += 1
    requirements_path = project_root / "requirements.txt"
    if requirements_path.exists():
        with open(requirements_path, 'r') as f:
            requirements = f.readlines()
        print(f"✓ requirements.txt exists ({len(requirements)} dependencies)")
        checks_passed += 1
    else:
        print("✗ requirements.txt not found")
    
    print("=" * 70)
    print(f"VALIDATION COMPLETE: {checks_passed}/{checks_total} checks passed")
    print("=" * 70)
    
    if checks_passed == checks_total:
        print("✓ All acceptance criteria met!")
        return 0
    else:
        print(f"⚠ {checks_total - checks_passed} check(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
