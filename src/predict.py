"""
Inference and Real-Time Power Prediction Module
Provides standalone CLI and API functions to predict CNC machining power consumption
from cutting tool, machine parameters, and workpiece material specifications.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import argparse
import joblib
import numpy as np
import pandas as pd
from src.data_preprocessing import add_engineered_features, get_all_feature_names

DEFAULT_MODEL_PATH = str(PROJECT_ROOT / "models" / "best_model.pkl")
DEFAULT_PIPELINE_PATH = str(PROJECT_ROOT / "models" / "preprocessing_pipeline.pkl")
DEFAULT_METADATA_PATH = str(PROJECT_ROOT / "models" / "feature_metadata.json")


def load_model_and_pipeline(
    model_path: str = DEFAULT_MODEL_PATH,
    pipeline_path: str = DEFAULT_PIPELINE_PATH
):
    """
    Loads saved champion model and preprocessing pipeline.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Pipeline file not found at: {pipeline_path}")
        
    model = joblib.load(model_path)
    pipeline = joblib.load(pipeline_path)
    return model, pipeline


def complete_machining_parameters(user_inputs: dict) -> dict:
    """
    Calculates dependent kinematic values (Spindle Speed, Feed Speed, MRR) if omitted.
    """
    d = user_inputs.copy()
    
    op = d.get("Operation_Type", "CNC Milling")
    vc = float(d.get("Cutting_Speed_vc_mpm", 150.0))
    diam = float(d.get("Tool_Diameter_mm", 16.0))
    flutes = int(d.get("Number_of_Flutes", 4 if op == "CNC Milling" else 1))
    f = float(d.get("Feed_Rate_mm_rev", 0.12))
    ap = float(d.get("Axial_Depth_ap_mm", 2.0))
    ae = float(d.get("Radial_Depth_ae_mm", 8.0 if op == "CNC Milling" else f))
    
    # Spindle Speed N = 1000 * vc / (pi * D)
    if "Spindle_Speed_RPM" not in d or d["Spindle_Speed_RPM"] is None:
        rpm = (1000.0 * vc) / (np.pi * max(diam, 1.0))
        d["Spindle_Speed_RPM"] = round(rpm, 1)
    else:
        rpm = float(d["Spindle_Speed_RPM"])
        
    # Feed speed vf
    if "Feed_Speed_vf_mmpm" not in d or d["Feed_Speed_vf_mmpm"] is None:
        if op == "CNC Milling":
            vf = f * flutes * rpm
        else:
            vf = f * rpm
        d["Feed_Speed_vf_mmpm"] = round(vf, 1)
    else:
        vf = float(d["Feed_Speed_vf_mmpm"])
        
    # Material Removal Rate MRR (cm3/min)
    if "Material_Removal_Rate_cm3_min" not in d or d["Material_Removal_Rate_cm3_min"] is None:
        if op == "CNC Milling":
            mrr = (ap * ae * vf) / 1000.0
        else:
            mrr = (vc * ap * f * 1000.0) / 1000.0
        d["Material_Removal_Rate_cm3_min"] = round(mrr, 3)
        
    if "Coolant_Flow_Rate_Lpm" not in d:
        coolant = d.get("Coolant_Condition", "Flood Coolant")
        flows = {"Dry": 0.0, "Flood Coolant": 25.0, "MQL (Min Lubrication)": 0.15, "Cryogenic (LN2)": 4.0}
        d["Coolant_Flow_Rate_Lpm"] = flows.get(coolant, 20.0)
        
    if "Material_Hardness_HB" not in d:
        hardness_defaults = {
            "AISI 1045 Steel": 210.0,
            "AISI 304 Stainless Steel": 200.0,
            "Ti-6Al-4V Titanium": 335.0,
            "Al 6061-T6 Aluminum": 95.0,
            "Inconel 718 Superalloy": 390.0
        }
        d["Material_Hardness_HB"] = hardness_defaults.get(d.get("Workpiece_Material"), 200.0)
        
    if "Tool_Wear_VB_mm" not in d:
        d["Tool_Wear_VB_mm"] = 0.10
        
    if "Rake_Angle_deg" not in d:
        d["Rake_Angle_deg"] = 6.0
        
    if "Tool_Coating" not in d:
        d["Tool_Coating"] = "TiAlN"
        
    return d


def predict_power(
    input_records,
    model=None,
    pipeline=None,
    model_path: str = DEFAULT_MODEL_PATH,
    pipeline_path: str = DEFAULT_PIPELINE_PATH
) -> np.ndarray:
    """
    Accepts single dictionary or list of dictionaries / pandas DataFrame,
    transforms features, and predicts electrical power consumption in kW.
    """
    if model is None or pipeline is None:
        model, pipeline = load_model_and_pipeline(model_path, pipeline_path)
        
    if isinstance(input_records, dict):
        df_input = pd.DataFrame([input_records])
    elif isinstance(input_records, list):
        df_input = pd.DataFrame(input_records)
    elif isinstance(input_records, pd.DataFrame):
        df_input = input_records.copy()
    else:
        raise TypeError("input_records must be dict, list of dicts, or pandas DataFrame")
        
    # Auto-fill missing kinematics for each record
    records = df_input.to_dict(orient="records")
    completed_records = [complete_machining_parameters(r) for r in records]
    df_completed = pd.DataFrame(completed_records)
    
    # Feature engineering
    df_feat = add_engineered_features(df_completed)
    
    # Transform
    X_trans = pipeline.transform(df_feat)
    
    # Load transformed feature names for clean DataFrame input
    if os.path.exists(DEFAULT_METADATA_PATH):
        try:
            with open(DEFAULT_METADATA_PATH, "r") as f:
                meta = json.load(f)
            feat_names = meta.get("transformed_feature_names")
            if feat_names and len(feat_names) == X_trans.shape[1]:
                X_trans = pd.DataFrame(X_trans, columns=feat_names)
        except Exception:
            pass
            
    # Predict
    preds = model.predict(X_trans)
    return np.clip(preds, 0.5, 50.0)


def main():
    parser = argparse.ArgumentParser(description="Predict CNC Machining Active Power Consumption")
    parser.add_argument("--operation", type=str, default="CNC Milling", choices=["CNC Milling", "CNC Turning"])
    parser.add_argument("--material", type=str, default="AISI 1045 Steel", help="Workpiece material")
    parser.add_argument("--vc", type=float, default=160.0, help="Cutting speed (m/min)")
    parser.add_argument("--feed", type=float, default=0.12, help="Feed rate (mm/rev or mm/tooth)")
    parser.add_argument("--ap", type=float, default=2.5, help="Axial depth of cut (mm)")
    parser.add_argument("--ae", type=float, default=10.0, help="Radial depth of cut (mm)")
    parser.add_argument("--diameter", type=float, default=16.0, help="Tool diameter (mm)")
    parser.add_argument("--flutes", type=int, default=4, help="Number of tool flutes")
    parser.add_argument("--wear", type=float, default=0.10, help="Tool flank wear VB (mm)")
    parser.add_argument("--coolant", type=str, default="Flood Coolant", help="Coolant condition")
    parser.add_argument("--coating", type=str, default="TiAlN", help="Tool coating")
    args = parser.parse_args()
    
    sample_input = {
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
    
    print("\n--- Input Machining Process Parameters ---")
    for k, v in sample_input.items():
        print(f"  {k:28s}: {v}")
        
    predicted_kw = predict_power(sample_input)[0]
    
    print("\n==========================================")
    print(f"  PREDICTED ACTIVE POWER:  {predicted_kw:.3f} kW")
    print("==========================================")
    
    # Energy rating
    if predicted_kw < 5.0:
        level = "Light Load (Eco/Finishing)"
    elif predicted_kw < 12.0:
        level = "Moderate Load (Standard Roughing/Semi-Finish)"
    else:
        level = "Heavy Load (High-Efficiency Roughing)"
    print(f"  Machine Load Status:     {level}\n")


if __name__ == "__main__":
    main()
