"""
Master Pipeline Runner: CNC Machining Power Consumption Prediction
Coordinates the complete machine learning engineering workflow:
  1. Data Ingestion & Validation (src.download_data)
  2. Physics Preprocessing & Feature Engineering (src.data_preprocessing)
  3. Model Training & 5-Fold Cross-Validation (src.train)
     - Linear Regression
     - Decision Tree Regressor
     - Random Forest Regressor
  4. Test Set Benchmarking & Diagnostic Plots (src.test)
  5. Real-Time Inference & Multi-Model Smoke Testing (src.load)
"""

# Import operating system interfaces
import os

# Import system-specific parameters and sys.path
import sys

# Import time module to record total execution duration
import time

# Import pathlib for clean path handling
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import pipeline components from src
from src.download_data import download_or_generate_dataset
from src.data_preprocessing import prepare_and_split_data
from src.train import load_training_and_testing_data, optimize_and_train_models, save_trained_models
from src.test import load_test_dataset, load_all_models, evaluate_model_on_test, generate_diagnostic_plots
from src.load import compare_all_models


def main():
    start_total_time = time.time()
    
    print("==================================================================")
    print("  CNC MACHINING POWER CONSUMPTION PREDICTION - MASTER PIPELINE   ")
    print("  Featuring: Linear Regression | Decision Tree | Random Forest   ")
    print("==================================================================")
    
    # -------------------------------------------------------------------------
    # STEP 1: Data Ingestion and Validation
    # -------------------------------------------------------------------------
    print("\n[Step 1/5] Ingesting and Validating 12,500 Machining Records...")
    raw_csv_path = os.path.join("data", "raw", "machining_power_consumption_12k.csv")
    df_raw = download_or_generate_dataset(
        target_filepath=raw_csv_path,
        sample_count=12500,
        random_seed=42,
        force_refresh=False
    )
    print(f"-> Verified dataset: {df_raw.shape[0]:,} rows and {df_raw.shape[1]} columns.")

    # -------------------------------------------------------------------------
    # STEP 2: Preprocessing and Feature Engineering
    # -------------------------------------------------------------------------
    print("\n[Step 2/5] Running Preprocessing and Multi-Physics Feature Engineering...")
    prepare_and_split_data(
        input_csv=raw_csv_path,
        output_dir="data/processed",
        models_dir="models"
    )
    print("-> Preprocessed train/test features and ColumnTransformer pipeline successfully saved.")

    # -------------------------------------------------------------------------
    # STEP 3: Model Training & Cross-Validation
    # -------------------------------------------------------------------------
    print("\n[Step 3/5] Training & Optimizing Core Models (Linear Regression, Decision Tree, Random Forest)...")
    X_train, X_test, y_train, y_test = load_training_and_testing_data()
    trained_artifacts = optimize_and_train_models(X_train, y_train, perform_cv=True)
    champion_name = save_trained_models(trained_artifacts, models_dir="models")
    print(f"-> Model training completed. Champion algorithm: {champion_name}")

    # -------------------------------------------------------------------------
    # STEP 4: Test Evaluation & Diagnostic Plots
    # -------------------------------------------------------------------------
    print("\n[Step 4/5] Evaluating on Unseen Test Set & Generating Diagnostic Plots...")
    loaded_models = load_all_models()
    test_results = {}
    for name, model in loaded_models.items():
        metrics, preds, residuals = evaluate_model_on_test(model, X_test, y_test)
        test_results[name] = {"metrics": metrics, "predictions": preds, "residuals": residuals}
        print(f"   [{name:18s}] R²: {metrics['R2']:.4f} | RMSE: {metrics['RMSE_kW']:.3f} kW | MAE: {metrics['MAE_kW']:.3f} kW | MAPE: {metrics['MAPE_percent']:.2f}%")
        
    generate_diagnostic_plots(test_results, y_test, figures_dir="reports/figures")
    print("-> Diagnostic evaluation plots generated in reports/figures/")

    # -------------------------------------------------------------------------
    # STEP 5: Multi-Model Inference Smoke Test
    # -------------------------------------------------------------------------
    print("\n[Step 5/5] Performing Multi-Model Live Inference Smoke Test...")
    test_scenario = {
        "Operation_Type": "CNC Milling",
        "Workpiece_Material": "Ti-6Al-4V Titanium",
        "Cutting_Speed_vc_mpm": 85.0,
        "Feed_Rate_mm_rev": 0.10,
        "Axial_Depth_ap_mm": 2.0,
        "Radial_Depth_ae_mm": 12.0,
        "Tool_Diameter_mm": 16.0,
        "Number_of_Flutes": 4,
        "Tool_Wear_VB_mm": 0.18,
        "Coolant_Condition": "Cryogenic (LN2)",
        "Tool_Coating": "AlCrN"
    }
    
    comparison_df = compare_all_models(test_scenario)
    print("\nLive Prediction Comparison for Ti-6Al-4V Cryogenic Milling:")
    print(comparison_df.to_string(index=False))

    total_duration = time.time() - start_total_time
    print("\n==================================================================")
    print(f"   MASTER PIPELINE COMPLETED SUCCESSFULLY IN {total_duration:.2f} SECONDS")
    print("==================================================================")


if __name__ == "__main__":
    main()
