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
        
    # Default cutting temperature in Celsius if omitted (derived via Boothroyd thermo-mechanical formula)
    if "Cutting_Temperature_C" not in d or d["Cutting_Temperature_C"] is None:
        clnt = d.get("Coolant_Condition", "Flood Coolant")
        mat = d.get("Workpiece_Material", "AISI 1045 Steel")
        coolant_temp_factor = {"Dry": 1.00, "Flood Coolant": 0.58, "MQL (Min Lubrication)": 0.80, "Cryogenic (LN2)": 0.32}.get(clnt, 0.65)
        mat_thermal_factor = {"Al 6061-T6 Aluminum": 0.50, "AISI 1045 Steel": 1.00, "AISI 304 Stainless Steel": 1.25, "Ti-6Al-4V Titanium": 1.55, "Inconel 718 Superalloy": 1.75}.get(mat, 1.0)
        vb = float(d.get("Tool_Wear_VB_mm", 0.10))
        wear_temp_boost = 1.0 + 1.25 * ((vb / 0.30) ** 1.15)
        t_est = 22.0 + (125.0 * ((vc / 100.0) ** 0.45) * ((f / 0.15) ** 0.22) * mat_thermal_factor * coolant_temp_factor * wear_temp_boost)
        d["Cutting_Temperature_C"] = round(float(np.clip(t_est, 45.0, 1100.0)), 1)

    return d


def get_tool_wear_condition(tool_wear_vb_mm: float) -> dict:
    """
    Evaluates tool wear land width (VB) against ISO 3685 failure threshold standards.
    Returns wear status description, severity level, and recommended maintenance action.
    """
    vb = float(tool_wear_vb_mm)
    if vb < 0.10:
        return {
            "Wear_Stage": "Initial / Break-in Wear",
            "VB_mm": vb,
            "Condition_Status": "Pristine / Healthy Tool",
            "ISO_Limit_Ratio_Percent": round(vb / 0.30 * 100.0, 1),
            "Action": "Continue Normal Machining"
        }
    elif vb <= 0.20:
        return {
            "Wear_Stage": "Steady-State Normal Wear",
            "VB_mm": vb,
            "Condition_Status": "Stable Flank Wear",
            "ISO_Limit_Ratio_Percent": round(vb / 0.30 * 100.0, 1),
            "Action": "Nominal Operation - Monitor Surface Finish"
        }
    elif vb <= 0.30:
        return {
            "Wear_Stage": "Accelerated / Severe Wear",
            "VB_mm": vb,
            "Condition_Status": "Warning - Approaching Limit",
            "ISO_Limit_Ratio_Percent": round(vb / 0.30 * 100.0, 1),
            "Action": "Schedule Tool Indexing / Edge Replacement Soon"
        }
    else:
        return {
            "Wear_Stage": "Critical Failure / Chipping Zone",
            "VB_mm": vb,
            "Condition_Status": "CRITICAL DANGER - Exceeds ISO 3685 Limit (0.30 mm)",
            "ISO_Limit_Ratio_Percent": round(vb / 0.30 * 100.0, 1),
            "Action": "STOP IMMEDIATELY - Replace Cutter Edge"
        }


# Standard Regional Electrical Grid Carbon Emission Factors (kg CO2e / kWh)
# Sources: International Energy Agency (IEA), US EPA eGRID, European Environment Agency (EEA), CEA India
GRID_CARBON_FACTORS = {
    "Global Average (0.475 kg/kWh)": 0.475,
    "US National Grid (0.385 kg/kWh)": 0.385,
    "European Union (0.255 kg/kWh)": 0.255,
    "China Grid (0.581 kg/kWh)": 0.581,
    "India Central Grid (0.708 kg/kWh)": 0.708,
    "100% Renewable / Solar (0.045 kg/kWh)": 0.045
}


def calculate_carbon_emissions(
    power_kw: float,
    coolant: str = "Flood Coolant",
    tool_wear_vb: float = 0.10,
    mrr: float = 10.0,
    grid_factor: float = 0.475,
    cut_time_sec: float = 60.0
) -> dict:
    """
    Computes comprehensive operational and life cycle carbon emissions (Scope 2 electricity,
    coolant fluid lifecycle, tool wear embodied carbon, specific carbon emission, and part footprint).
    """
    # Electrical Scope 2 carbon emission rate (kg CO2e / hr)
    ce_electrical_rate = power_kw * grid_factor
    
    # Cutting fluid lifecycle emission rate (kg CO2e / hr)
    fluid_emissions = {
        "Dry": 0.00,
        "MQL (Min Lubrication)": 0.08,
        "Flood Coolant": 0.45,
        "Cryogenic (LN2)": 0.65
    }
    ce_fluid_rate = fluid_emissions.get(coolant, 0.35)
    
    # Tool insert embodied carbon rate (kg CO2e / hr) based on wear progression
    ce_tool_rate = 0.05 * (1.0 + 1.25 * ((tool_wear_vb / 0.30) ** 1.15))
    
    # Total operational carbon emission rate (kg CO2e / hr)
    total_ce_rate = ce_electrical_rate + ce_fluid_rate + ce_tool_rate
    
    # Per-part carbon emissions
    part_hours = cut_time_sec / 3600.0
    part_co2_kg = total_ce_rate * part_hours
    part_co2_g = part_co2_kg * 1000.0
    
    # Specific Carbon Emission (SCE in g CO2e / cm3 of material removed)
    mrr_cm3_hr = max(mrr * 60.0, 0.01)
    sce_g_cm3 = (total_ce_rate * 1000.0) / mrr_cm3_hr
    
    # ESG Scope 2 rating category
    if total_ce_rate < 1.5:
        esg_rating = "Eco-Optimized (Green Tier)"
        esg_badge = "[Low Carbon - Green]"
    elif total_ce_rate <= 4.0:
        esg_rating = "Moderate Footprint (Amber Tier)"
        esg_badge = "[Moderate - Amber]"
    else:
        esg_rating = "Carbon-Intensive (Red Tier)"
        esg_badge = "[High Carbon - Red]"
        
    return {
        "Grid_Factor_kg_kWh": grid_factor,
        "Electrical_Carbon_Rate_kg_hr": round(ce_electrical_rate, 4),
        "Coolant_Carbon_Rate_kg_hr": round(ce_fluid_rate, 4),
        "Fluid_Carbon_Rate_kg_hr": round(ce_fluid_rate, 4),
        "Tool_Embodied_Carbon_Rate_kg_hr": round(ce_tool_rate, 4),
        "Tool_Wear_Carbon_Rate_kg_hr": round(ce_tool_rate, 4),
        "Total_Carbon_Rate_kg_hr": round(total_ce_rate, 4),
        "Per_Part_Carbon_g": round(part_co2_g, 2),
        "Part_Carbon_Footprint_g": round(part_co2_g, 2),
        "Per_Part_Carbon_kg": round(part_co2_kg, 4),
        "Part_Carbon_Footprint_kg": round(part_co2_kg, 4),
        "Specific_Carbon_Emission_g_cm3": round(sce_g_cm3, 3),
        "ESG_Rating": esg_rating,
        "ESG_Badge": esg_badge
    }


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


def predict_carbon_emission(
    input_data,
    grid_factor: float = 0.475,
    cut_time_sec: float = 60.0,
    model_name: str = "best"
) -> dict:
    """
    Predicts active electrical power and calculates complete operational carbon emissions.
    """
    pred_kw = float(predict_power(input_data, model_name=model_name)[0])
    
    # Extract process attributes for full lifecycle carbon accounting
    if isinstance(input_data, dict):
        rec = input_data
    elif isinstance(input_data, pd.DataFrame):
        rec = input_data.iloc[0].to_dict()
    elif isinstance(input_data, list):
        rec = input_data[0]
    else:
        rec = {}
        
    clnt = rec.get("Coolant_Condition", "Flood Coolant")
    vb = float(rec.get("Tool_Wear_VB_mm", 0.10))
    mrr = float(rec.get("Material_Removal_Rate_cm3_min", 12.0))
    
    ce_metrics = calculate_carbon_emissions(
        power_kw=pred_kw,
        coolant=clnt,
        tool_wear_vb=vb,
        mrr=mrr,
        grid_factor=grid_factor,
        cut_time_sec=cut_time_sec
    )
    ce_metrics["Predicted_Power_kW"] = round(pred_kw, 3)
    return ce_metrics


def compare_all_models(input_record: dict, grid_factor: float = 0.475) -> pd.DataFrame:
    """
    Runs the given input record through all three models and produces a multi-model
    comparison table displaying both Predicted Power (kW) and Carbon Emission Rate (kg CO2e/hr).
    """
    # Load pipeline
    pipeline = load_preprocessing_pipeline()
    
    # Load all 3 models
    all_models = load_all_trained_models()
    
    # Extract process attributes
    clnt = input_record.get("Coolant_Condition", "Flood Coolant")
    vb = float(input_record.get("Tool_Wear_VB_mm", 0.10))
    mrr = float(input_record.get("Material_Removal_Rate_cm3_min", 12.0))
    
    # Dictionary to collect predictions
    preds_summary = []
    
    # Iterate over available models
    for key, model_obj in all_models.items():
        formatted_name = key.replace("_", " ").title()
        pred_kw = predict_power(input_record, model=model_obj, pipeline=pipeline)[0]
        ce_calc = calculate_carbon_emissions(float(pred_kw), coolant=clnt, tool_wear_vb=vb, mrr=mrr, grid_factor=grid_factor)
        
        preds_summary.append({
            "Model Key": key,
            "Algorithm": formatted_name,
            "Predicted Power (kW)": round(float(pred_kw), 3),
            "Carbon Rate (kg CO2e/hr)": ce_calc["Total_Carbon_Rate_kg_hr"],
            "Specific Carbon (g/cm3)": ce_calc["Specific_Carbon_Emission_g_cm3"],
            "ESG Tier": ce_calc["ESG_Badge"]
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
    parser.add_argument("--temp", type=float, default=None, help="Cutting zone temperature in C (auto-estimated if omitted)")
    parser.add_argument("--grid", type=float, default=0.475, help="Grid emission factor in kg CO2e/kWh (default: 0.475 Global Average)")
    parser.add_argument("--cut-time", type=float, default=60.0, help="Cutting cycle time per part in seconds (default: 60.0)")
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
    if args.temp is not None:
        sample["Cutting_Temperature_C"] = args.temp
    
    print("=" * 70)
    print("      CNC MACHINING POWER & CARBON PREDICTION (src/load.py)")
    print("=" * 70)
    print("\nInput Machining Parameters:")
    for k, v in sample.items():
        print(f"  {k:28s}: {v}")
        
    # Tool Wear Health & Diagnostic Analysis
    wear_diag = get_tool_wear_condition(args.wear)
    print("\n--- Tool Wear Diagnostic Assessment (ISO 3685) ---")
    print(f"  Flank Wear Land (VB)        : {wear_diag['VB_mm']:.3f} mm")
    print(f"  Degradation Stage           : {wear_diag['Wear_Stage']}")
    print(f"  Health Condition Status     : {wear_diag['Condition_Status']}")
    print(f"  ISO 0.30mm Limit Utilized   : {wear_diag['ISO_Limit_Ratio_Percent']:.1f} %")
    print(f"  Recommended Shop Action     : {wear_diag['Action']}")
        
    if args.model == "compare":
        print("\n" + "-" * 70)
        print("     MULTI-MODEL POWER & CARBON EMISSION PREDICTION COMPARISON")
        print("-" * 70)
        comparison_df = compare_all_models(sample, grid_factor=args.grid)
        print(comparison_df.to_string(index=False))
        print("-" * 70)
    else:
        ce_res = predict_carbon_emission(sample, grid_factor=args.grid, cut_time_sec=args.cut_time, model_name=args.model)
        print("\n" + "=" * 55)
        print(f"  Model Used                 : {args.model}")
        print(f"  Predicted Power Demand     : {ce_res['Predicted_Power_kW']:.3f} kW")
        print(f"  Total Carbon Emission Rate : {ce_res['Total_Carbon_Rate_kg_hr']:.4f} kg CO2e/hr")
        print(f"  Specific Carbon Emission   : {ce_res['Specific_Carbon_Emission_g_cm3']:.3f} g CO2e/cm3")
        print(f"  Carbon Footprint per Part  : {ce_res['Per_Part_Carbon_g']:.2f} g CO2e/part ({args.cut_time}s cut)")
        print(f"  Scope 2 ESG Rating         : {ce_res['ESG_Rating']}")
        print("=" * 55)
        
    print("\n[OK] Inference finished successfully.")


# Execute when script is run directly
if __name__ == "__main__":
    main()
