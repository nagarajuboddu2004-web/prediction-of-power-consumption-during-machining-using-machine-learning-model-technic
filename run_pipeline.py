"""
Master Pipeline Runner: Machining Power Consumption Prediction
Executes the full workflow:
  1. Physics-based Dataset Generation (>12k records)
  2. Data Preprocessing & Feature Engineering
  3. Model Training & Cross-Benchmarking
  4. Diagnostic Evaluation & Plot Generation
  5. Inference Smoke Test Verification
"""

import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_generator import generate_machining_dataset
from src.data_preprocessing import prepare_and_split_data
from src.model_training import train_and_benchmark
from src.model_evaluation import generate_evaluation_report
from src.predict import predict_power


def main():
    start_time = time.time()
    print("==================================================================")
    print("  CNC MACHINING POWER CONSUMPTION PREDICTION - ML PIPELINE        ")
    print("==================================================================")
    
    # 1. Dataset Generation
    dataset_path = "data/raw/machining_power_consumption_12k.csv"
    if not os.path.exists(dataset_path):
        print("\n[Step 1/5] Generating 12,500 physics-grounded machining observations...")
        os.makedirs("data/raw", exist_ok=True)
        df_data = generate_machining_dataset(num_samples=12500, random_seed=42)
        df_data.to_csv(dataset_path, index=False)
        print(f"Generated dataset with {df_data.shape[0]} rows and {df_data.shape[1]} columns at: {dataset_path}")
    else:
        print(f"\n[Step 1/5] Dataset found at {dataset_path}. Skipping generation.")

    # 2. Data Preprocessing & Feature Engineering
    print("\n[Step 2/5] Running preprocessing, engineering features, and splitting data...")
    prepare_and_split_data(
        input_csv=dataset_path,
        output_dir="data/processed",
        models_dir="models"
    )

    # 3. Model Training & Benchmarking
    print("\n[Step 3/5] Training candidate ML algorithms & evaluating performance...")
    best_model_name, leaderboard_df = train_and_benchmark(models_dir="models")

    # 4. Model Evaluation & Visualization
    print("\n[Step 4/5] Generating diagnostic plots and evaluation report...")
    generate_evaluation_report(models_dir="models", reports_dir="reports")

    # 5. Inference Smoke Test
    print("\n[Step 5/5] Performing live inference smoke test...")
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
    pred_kw = predict_power(test_scenario)[0]
    print(f"   Test Scenario Material: {test_scenario['Workpiece_Material']}")
    print(f"   Cutting Speed:          {test_scenario['Cutting_Speed_vc_mpm']} m/min")
    print(f"   Predicted Active Power: {pred_kw:.3f} kW")

    total_duration = time.time() - start_time
    print("\n==================================================================")
    print(f"   PIPELINE COMPLETE SUCCESSFULLY IN {total_duration:.1f} SECONDS")
    print("==================================================================")
    print("To launch the interactive Web Dashboard:")
    print("   streamlit run app/streamlit_app.py")
    print("==================================================================")


if __name__ == "__main__":
    main()
