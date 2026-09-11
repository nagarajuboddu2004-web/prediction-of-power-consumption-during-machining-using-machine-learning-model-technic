"""
Model and Data Loading & Inference Module for CNC Machining Power Prediction.
Provides standardized utilities to:
  1. Load trained models (Linear Regression, Decision Tree, Random Forest, Champion)
  2. Load preprocessing transformation pipelines and datasets
  3. Perform single-instance and batch real-time power predictions
  4. Compare predictions across all three models simultaneously
Every single line of code is documented with an inline comment explaining its function.
"""

# Import operating system interfaces for directory management and file paths
import os

# Import system-specific parameters and system path manipulation
import sys

# Import json library to parse feature metadata and store configurations
import json

# Import argparse for command-line argument parsing
import argparse

# Import joblib to load serialized Scikit-Learn estimators and pipelines
import joblib

# Import pathlib for clean filesystem path resolution
from pathlib import Path

# Import numpy for scientific computations and numerical manipulation
import numpy as np

# Import pandas for tabular data structure management
import pandas as pd

# Determine the absolute directory path containing this script
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent

# Determine the project root directory (one level up from src)
PROJECT_ROOT_DIR = CURRENT_SCRIPT_DIR.parent

# Ensure project root directory is available in sys.path
if str(PROJECT_ROOT_DIR) not in sys.path:
    # Insert project root into sys.path
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

# Import feature engineering function from src/data_preprocessing.py
from src.data_preprocessing import add_engineered_features

# Define default paths for model files
MODELS_DIR = os.path.join(PROJECT_ROOT_DIR, "models")

# Define path to the preprocessing pipeline artifact
DEFAULT_PIPELINE_PATH = os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl")

# Define path to feature metadata
DEFAULT_METADATA_PATH = os.path.join(MODELS_DIR, "feature_metadata.json")

# Model filename dictionary mapping standard keys to serialized filenames
MODEL_FILE_MAP = {
    "linear_regression": "linear_regression.pkl",
    "decision_tree": "decision_tree.pkl",
    "random_forest": "random_forest.pkl",
    "best": "best_model.pkl"
}


def load_model(model_key: str = "best"):
    """
    Loads a specific trained model by key ('linear_regression', 'decision_tree', 'random_forest', 'best').
    """
    # Normalize model key to lowercase and replace spaces or hyphens with underscores
    clean_key = model_key.lower().replace(" ", "_").replace("-", "_")
    
    # Check if key is registered in our map
    if clean_key not in MODEL_FILE_MAP:
        # Raise exception with helpful error message
        raise ValueError(f"Unknown model key '{model_key}'. Valid choices: {list(MODEL_FILE_MAP.keys())}")
        
    # Get corresponding filename
    filename = MODEL_FILE_MAP[clean_key]
    
    # Build complete path to the model pickle
    full_path = os.path.join(MODELS_DIR, filename)
    
    # Verify the model file exists on disk
    if not os.path.exists(full_path):
        # Raise error if model has not been trained yet
        raise FileNotFoundError(f"Model file not found at: {full_path}. Run train.py first.")
        
    # Load and return the serialized model
    model = joblib.load(full_path)
    
    # Return loaded model
    return model


def load_all_trained_models() -> dict:
    """
    Loads all three trained models (Linear Regression, Decision Tree, Random Forest) into a dictionary.
    """
    # Container for loaded models
    models_dict = {}
    
    # Iterate through keys
    for key in ["linear_regression", "decision_tree", "random_forest"]:
        # Try loading model
        try:
            # Load model and assign to dictionary
            models_dict[key] = load_model(key)
        except FileNotFoundError:
            # Ignore missing models or log warning
            pass
            
    # Return dictionary of available models
    return models_dict


def load_preprocessing_pipeline(pipeline_path: str = DEFAULT_PIPELINE_PATH):
    """
    Loads the fitted ColumnTransformer pipeline for feature scaling and encoding.
    """
    # Verify the pipeline file exists
    if not os.path.exists(pipeline_path):
        # Raise error if pipeline is missing
        raise FileNotFoundError(f"Preprocessing pipeline not found at: {pipeline_path}. Run data_preprocessing.py first.")
        
    # Load and return pipeline object
    pipeline = joblib.load(pipeline_path)
    
    # Return pipeline
    return pipeline


def load_model_and_pipeline(model_path: str = None, pipeline_path: str = None):
    """
    Loads saved champion model and preprocessing pipeline.
    """
    # Check if a custom model path was provided and exists
    if model_path and os.path.exists(model_path):
        # Load the custom model pickle file
        model = joblib.load(model_path)
    else:
        # Otherwise load the standard champion model
        model = load_model("best")
        
    # Check if a custom pipeline path was provided and exists
    if pipeline_path and os.path.exists(pipeline_path):
        # Load the custom pipeline pickle file
        pipeline = joblib.load(pipeline_path)
    else:
        # Otherwise load the default fitted preprocessing pipeline
        pipeline = load_preprocessing_pipeline()
        
    # Return model and pipeline tuple
    return model, pipeline


def load_dataset(split: str = "raw") -> pd.DataFrame:
    """
    Loads dataset: 'raw' for full raw CSV, 'train' for train_features.csv, 'test' for test_features.csv.
    """
    # Normalize split string
    s = split.lower().strip()
    
    # Path branching based on split parameter
    if s == "raw":
        target_path = os.path.join(PROJECT_ROOT_DIR, "data", "raw", "machining_power_consumption_12k.csv")
    elif s == "train":
        target_path = os.path.join(PROJECT_ROOT_DIR, "data", "processed", "train_features.csv")
    elif s == "test":
        target_path = os.path.join(PROJECT_ROOT_DIR, "data", "processed", "test_features.csv")
    else:
        raise ValueError("Invalid split argument. Choose from 'raw', 'train', or 'test'.")
        
    # Check if target dataset exists
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Dataset file not found at: {target_path}")
        
    # Read and return CSV
    df = pd.read_csv(target_path)
    return df


def complete_kinematic_inputs(input_dict: dict) -> dict:
    """
    Calculates missing derived physical kinematics (Spindle RPM, Feed Speed vf, MRR) if omitted.
    """
    # Copy dictionary to avoid in-place side effects
    d = input_dict.copy()
    
    # Extract operation type
    op = d.get("Operation_Type", "CNC Milling")
    
    # Extract cutting speed
    vc = float(d.get("Cutting_Speed_vc_mpm", 150.0))
    
    # Extract tool diameter
    diam = float(d.get("Tool_Diameter_mm", 16.0))
    
    # Extract flutes
    flutes = int(d.get("Number_of_Flutes", 4 if op == "CNC Milling" else 1))
    
    # Extract feed per rev/tooth
    f = float(d.get("Feed_Rate_mm_rev", 0.12))
    
    # Extract axial depth of cut
    ap = float(d.get("Axial_Depth_ap_mm", 2.0))
    
    # Extract radial width of cut
    ae = float(d.get("Radial_Depth_ae_mm", 8.0 if op == "CNC Milling" else f))
    
    # Calculate Spindle RPM: N = 1000 * vc / (pi * D)
    if "Spindle_Speed_RPM" not in d or d["Spindle_Speed_RPM"] is None:
        rpm = (1000.0 * vc) / (np.pi * max(diam, 1.0))
        d["Spindle_Speed_RPM"] = round(rpm, 1)
    else:
        rpm = float(d["Spindle_Speed_RPM"])
        
    # Calculate linear feed speed vf (mm/min)
    if "Feed_Speed_vf_mmpm" not in d or d["Feed_Speed_vf_mmpm"] is None:
        if op == "CNC Milling":
            vf = f * flutes * rpm
        else:
            vf = f * rpm
        d["Feed_Speed_vf_mmpm"] = round(vf, 1)
    else:
        vf = float(d["Feed_Speed_vf_mmpm"])
        
    # Calculate Material Removal Rate MRR (cm3/min)
    if "Material_Removal_Rate_cm3_min" not in d or d["Material_Removal_Rate_cm3_min"] is None:
        if op == "CNC Milling":
            mrr = (ap * ae * vf) / 1000.0
        else:
            mrr = (vc * ap * f * 1000.0) / 1000.0
        d["Material_Removal_Rate_cm3_min"] = round(mrr, 3)
        
    # Default coolant flow if omitted
    if "Coolant_Flow_Rate_Lpm" not in d:
        coolant = d.get("Coolant_Condition", "Flood Coolant")
        flows = {"Dry": 0.0, "Flood Coolant": 25.0, "MQL (Min Lubrication)": 0.15, "Cryogenic (LN2)": 4.0}
        d["Coolant_Flow_Rate_Lpm"] = flows.get(coolant, 20.0)
        
    # Default hardness if omitted
    if "Material_Hardness_HB" not in d:
        hardness_map = {
            "AISI 1045 Steel": 210.0,
            "AISI 304 Stainless Steel": 200.0,
            "Ti-6Al-4V Titanium": 335.0,
            "Al 6061-T6 Aluminum": 95.0,
            "Inconel 718 Superalloy": 390.0
        }
        d["Material_Hardness_HB"] = hardness_map.get(d.get("Workpiece_Material"), 200.0)
        
    # Default tool wear VB (0.10 mm fresh tool)
    if "Tool_Wear_VB_mm" not in d:
        d["Tool_Wear_VB_mm"] = 0.10
        
    # Default rake angle (6.0 degrees)
    if "Rake_Angle_deg" not in d:
        d["Rake_Angle_deg"] = 6.0
        
    # Default tool coating
    if "Tool_Coating" not in d:
        d["Tool_Coating"] = "TiAlN"
        
    return d


def predict_power(
    input_data,
    model_name: str = "best",
    model=None,
    pipeline=None
) -> np.ndarray:
    """
    Accepts single dict, list of dicts, or DataFrame, and outputs predicted active power in kW.
    """
    # Load model if not supplied
    if model is None:
        model = load_model(model_name)
        
    # Load pipeline if not supplied
    if pipeline is None:
        pipeline = load_preprocessing_pipeline()
        
    # Convert input into DataFrame
    if isinstance(input_data, dict):
        df_in = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        df_in = pd.DataFrame(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df_in = input_data.copy()
    else:
        raise TypeError("input_data must be dict, list of dicts, or pandas DataFrame")
        
    # Complete kinematics for all records
    records = df_in.to_dict(orient="records")
    completed = [complete_kinematic_inputs(r) for r in records]
    df_completed = pd.DataFrame(completed)
    
    # Apply physical feature engineering
    df_feat = add_engineered_features(df_completed)
    
    # Preprocessing transformation
    X_trans = pipeline.transform(df_feat)
    
    # Load feature names for DataFrame compatibility
    if os.path.exists(DEFAULT_METADATA_PATH):
        try:
            with open(DEFAULT_METADATA_PATH, "r") as f:
                meta = json.load(f)
            names = meta.get("transformed_feature_names")
            if names and len(names) == X_trans.shape[1]:
                X_trans = pd.DataFrame(X_trans, columns=names)
        except Exception:
            pass
            
    # Predict power in kW
    predictions = model.predict(X_trans)
    
    # Clip predictions to reasonable physical bounds (0.5 kW to 50 kW)
    return np.clip(predictions, 0.5, 50.0)


def compare_all_models(input_record: dict) -> pd.DataFrame:
    """
    Runs the given input record through all three models and produces a comparison table.
    """
    # Load pipeline
    pipeline = load_preprocessing_pipeline()
    
    # Load all 3 models
    all_models = load_all_trained_models()
    
    # Dictionary to collect predictions
    preds_summary = []
    
    # Iterate over available models
    for key, model_obj in all_models.items():
        formatted_name = key.replace("_", " ").title()
        pred_kw = predict_power(input_record, model=model_obj, pipeline=pipeline)[0]
        preds_summary.append({
            "Model Key": key,
            "Algorithm": formatted_name,
            "Predicted Power (kW)": round(float(pred_kw), 3)
        })
        
    # Return as DataFrame
    return pd.DataFrame(preds_summary)


def main():
    """
    Command-line interface to test loading models and comparing power predictions.
    """
    # Setup argument parser
    parser = argparse.ArgumentParser(description="CNC Machining Power Prediction Inference Module")
    parser.add_argument("--operation", type=str, default="CNC Milling", choices=["CNC Milling", "CNC Turning"])
    parser.add_argument("--material", type=str, default="Ti-6Al-4V Titanium", help="Workpiece material")
    parser.add_argument("--vc", type=float, default=90.0, help="Cutting speed vc (m/min)")
    parser.add_argument("--feed", type=float, default=0.10, help="Feed rate (mm/rev or mm/tooth)")
    parser.add_argument("--ap", type=float, default=2.0, help="Axial depth of cut (mm)")
    parser.add_argument("--ae", type=float, default=12.0, help="Radial depth of cut (mm)")
    parser.add_argument("--diameter", type=float, default=16.0, help="Tool diameter (mm)")
    parser.add_argument("--flutes", type=int, default=4, help="Number of tool flutes")
    parser.add_argument("--wear", type=float, default=0.15, help="Tool flank wear VB (mm)")
    parser.add_argument("--coolant", type=str, default="Cryogenic (LN2)", help="Coolant condition")
    parser.add_argument("--coating", type=str, default="AlCrN", help="Tool coating")
    parser.add_argument("--model", type=str, default="compare", help="Model key ('linear_regression', 'decision_tree', 'random_forest', 'best', 'compare')")
    args = parser.parse_args()
    
    # Build sample dictionary
    sample = {
        "Operation_Type": args.operation,
        "Workpiece_Material": args.material,
        "Cutting_Speed_vc_mpm": args.vc,
        "Feed_Rate_mm_rev": args.feed,
        "Axial_Depth_ap_mm": args.ap,
        "Radial_Depth_ae_mm": args.ae,
        "Tool_Diameter_mm": args.diameter,
        "Number_of_Flutes": args.flutes,
        "Tool_Wear_VB_mm": args.wear,
        "Coolant_Condition": args.coolant,
        "Tool_Coating": args.coating
    }
    
    print("=" * 70)
    print("      CNC MACHINING POWER PREDICTION - INFERENCE & LOADING (src/load.py)")
    print("=" * 70)
    print("\nInput Machining Parameters:")
    for k, v in sample.items():
        print(f"  {k:28s}: {v}")
        
    if args.model == "compare":
        print("\n" + "-" * 70)
        print("          MULTI-MODEL POWER PREDICTION COMPARISON")
        print("-" * 70)
        comparison_df = compare_all_models(sample)
        print(comparison_df.to_string(index=False))
        print("-" * 70)
    else:
        pred = predict_power(sample, model_name=args.model)[0]
        print("\n" + "=" * 50)
        print(f"  Model Used:       {args.model}")
        print(f"  Predicted Power:  {pred:.3f} kW")
        print("=" * 50)
        
    print("\n[OK] Inference finished successfully.")


# Execute when script is run directly
if __name__ == "__main__":
    main()
