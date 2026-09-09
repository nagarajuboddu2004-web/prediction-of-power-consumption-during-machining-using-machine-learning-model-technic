"""
Data Preprocessing and Feature Engineering Module
Prepares raw machining data for model training, builds transformation pipelines,
and prevents data leakage by isolating input kinematic/material parameters from targets.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Base predictive features known prior to machining
CATEGORICAL_FEATURES = [
    "Operation_Type",
    "Workpiece_Material",
    "Tool_Coating",
    "Coolant_Condition"
]

NUMERICAL_FEATURES = [
    "Material_Hardness_HB",
    "Tool_Diameter_mm",
    "Number_of_Flutes",
    "Tool_Wear_VB_mm",
    "Rake_Angle_deg",
    "Cutting_Speed_vc_mpm",
    "Spindle_Speed_RPM",
    "Feed_Rate_mm_rev",
    "Axial_Depth_ap_mm",
    "Radial_Depth_ae_mm",
    "Feed_Speed_vf_mmpm",
    "Coolant_Flow_Rate_Lpm",
    "Material_Removal_Rate_cm3_min"
]

TARGET_COLUMN = "Power_Consumption_kW"


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates physically informed interaction features.
    """
    df = df.copy()
    
    # 1. Kinematic Interaction: Cutting Speed x Feed Rate
    df["Speed_x_Feed"] = df["Cutting_Speed_vc_mpm"] * df["Feed_Rate_mm_rev"]
    
    # 2. Geometry: Depth ratio (ap / ae)
    df["Depth_Ratio"] = df["Axial_Depth_ap_mm"] / (df["Radial_Depth_ae_mm"] + 1e-4)
    
    # 3. Dynamic Load per flute
    safe_flutes = df["Number_of_Flutes"].clip(lower=1)
    df["MRR_per_Flute"] = df["Material_Removal_Rate_cm3_min"] / safe_flutes
    
    # 4. Normalized Tool Wear relative to tool diameter
    df["Wear_to_Diameter_Ratio"] = df["Tool_Wear_VB_mm"] / (df["Tool_Diameter_mm"] + 1e-4)
    
    return df


def get_all_feature_names():
    """
    Returns list of all feature names including engineered features.
    """
    engineered_numerical = [
        "Speed_x_Feed",
        "Depth_Ratio",
        "MRR_per_Flute",
        "Wear_to_Diameter_Ratio"
    ]
    all_numerical = NUMERICAL_FEATURES + engineered_numerical
    return CATEGORICAL_FEATURES, all_numerical


def build_preprocessor():
    """
    Builds a Scikit-Learn ColumnTransformer for numerical scaling and categorical encoding.
    """
    cat_cols, num_cols = get_all_feature_names()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ],
        remainder="drop"
    )
    return preprocessor


def prepare_and_split_data(
    input_csv: str = "data/raw/machining_power_consumption_12k.csv",
    output_dir: str = "data/processed",
    models_dir: str = "models",
    test_size: float = 0.20,
    random_state: int = 42
):
    """
    Loads raw data, performs feature engineering, fits preprocessing pipeline,
    and saves train/test sets and fitted transformers.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"Loading raw dataset from {input_csv}...")
    df_raw = pd.read_csv(input_csv)
    
    # Validate target exists
    if TARGET_COLUMN not in df_raw.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")
        
    print(f"Original shape: {df_raw.shape}")
    
    # Add engineered features
    df_feat = add_engineered_features(df_raw)
    
    cat_cols, num_cols = get_all_feature_names()
    feature_cols = cat_cols + num_cols
    
    X = df_feat[feature_cols]
    y = df_feat[TARGET_COLUMN]
    
    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Fitting ColumnTransformer on {X_train.shape[0]} training samples...")
    preprocessor = build_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    # Extract feature names after OneHotEncoding
    ohe = preprocessor.named_transformers_["cat"]
    cat_encoded_names = list(ohe.get_feature_names_out(cat_cols))
    transformed_feature_names = num_cols + cat_encoded_names
    
    # Save processed numpy / csv
    train_df_processed = pd.DataFrame(X_train_trans, columns=transformed_feature_names)
    train_df_processed[TARGET_COLUMN] = y_train.values
    train_df_processed.to_csv(os.path.join(output_dir, "train_features.csv"), index=False)
    
    test_df_processed = pd.DataFrame(X_test_trans, columns=transformed_feature_names)
    test_df_processed[TARGET_COLUMN] = y_test.values
    test_df_processed.to_csv(os.path.join(output_dir, "test_features.csv"), index=False)
    
    # Save preprocessor
    pipeline_path = os.path.join(models_dir, "preprocessing_pipeline.pkl")
    joblib.dump(preprocessor, pipeline_path)
    print(f"Saved preprocessing pipeline to {pipeline_path}")
    
    # Save feature metadata
    metadata = {
        "raw_categorical_features": cat_cols,
        "raw_numerical_features": NUMERICAL_FEATURES,
        "engineered_features": ["Speed_x_Feed", "Depth_Ratio", "MRR_per_Flute", "Wear_to_Diameter_Ratio"],
        "all_input_features": feature_cols,
        "transformed_feature_names": transformed_feature_names,
        "target_column": TARGET_COLUMN,
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }
    metadata_path = os.path.join(models_dir, "feature_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Metadata saved to {metadata_path}")
    print(f"Data preparation complete. Processed {len(transformed_feature_names)} features.")
    
    return X_train, X_test, y_train, y_test, preprocessor, transformed_feature_names


if __name__ == "__main__":
    prepare_and_split_data()
