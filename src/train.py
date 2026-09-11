"""
Model Training, Cross-Validation, and Hyperparameter Optimization Module.
Trains three core machine learning algorithms for CNC machining power prediction:
  1. Linear Regression (Baseline Parametric Model)
  2. Decision Tree Regressor (Non-Linear Single-Tree Estimator)
  3. Random Forest Regressor (Ensemble Bagging Estimator)
Every single line of code is documented with an inline comment explaining its function.
"""

# Import operating system interfaces for path handling and file operations
import os

# Import system-specific parameters to manipulate Python module search paths
import sys

# Import time module to benchmark algorithm training execution durations
import time

# Import json library to serialize model metadata and benchmark metrics
import json

# Import joblib for robust serialization and persistence of trained Scikit-Learn estimators
import joblib

# Import pathlib for object-oriented filesystem path resolution
from pathlib import Path

# Import numpy for high-performance numerical operations and matrix calculations
import numpy as np

# Import pandas for structured DataFrame handling and tabular data manipulation
import pandas as pd

# Import Linear Regression from Scikit-Learn linear model subpackage
from sklearn.linear_model import LinearRegression

# Import Decision Tree Regressor from Scikit-Learn tree subpackage
from sklearn.tree import DecisionTreeRegressor

# Import Random Forest Regressor from Scikit-Learn ensemble subpackage
from sklearn.ensemble import RandomForestRegressor

# Import KFold and cross_val_score for robust k-fold cross-validation
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

# Import standard regression evaluation metrics from Scikit-Learn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# Determine the directory location of this script file
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent

# Determine the root directory of the project
PROJECT_ROOT_DIR = CURRENT_SCRIPT_DIR.parent

# Ensure the project root is in sys.path so modules can be imported
if str(PROJECT_ROOT_DIR) not in sys.path:
    # Prepend project root directory to sys.path
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

# Import data preparation routine from src/data_preprocessing.py
from src.data_preprocessing import prepare_and_split_data

# Define standard target column name for active electrical power
TARGET_COLUMN = "Power_Consumption_kW"


def load_training_and_testing_data(
    processed_dir: str = "data/processed",
    models_dir: str = "models",
    raw_csv_path: str = "data/raw/machining_power_consumption_12k.csv"
):
    """
    Loads preprocessed feature matrices and target vectors.
    If preprocessed files do not exist, triggers automatic preprocessing pipeline.
    """
    # Build the full file path for the preprocessed training features CSV
    train_features_path = os.path.join(PROJECT_ROOT_DIR, processed_dir, "train_features.csv")
    
    # Build the full file path for the preprocessed testing features CSV
    test_features_path = os.path.join(PROJECT_ROOT_DIR, processed_dir, "test_features.csv")
    
    # Check if either training or testing preprocessed file is missing
    if not os.path.exists(train_features_path) or not os.path.exists(test_features_path):
        # Notify the user that preprocessing pipeline is being automatically triggered
        print("[INFO] Preprocessed feature sets not found. Running prepare_and_split_data()...")
        
        # Resolve the full path to the raw dataset CSV
        full_raw_path = os.path.join(PROJECT_ROOT_DIR, raw_csv_path)
        
        # Execute preprocessing and feature engineering pipeline to generate feature files
        prepare_and_split_data(
            input_csv=full_raw_path,
            output_dir=os.path.join(PROJECT_ROOT_DIR, processed_dir),
            models_dir=os.path.join(PROJECT_ROOT_DIR, models_dir)
        )
    
    # Read training features DataFrame from disk
    train_df = pd.read_csv(train_features_path)
    
    # Read testing features DataFrame from disk
    test_df = pd.read_csv(test_features_path)
    
    # Separate input training features by dropping target columns to avoid data leakage
    target_cols = [TARGET_COLUMN, "Carbon_Emission_Rate_kgCO2e_hr", "Specific_Carbon_Emission_gCO2e_cm3"]
    cols_to_drop = [c for c in target_cols if c in train_df.columns]
    
    # Input feature matrix X_train (strictly 35 features)
    X_train = train_df.drop(columns=cols_to_drop)
    
    # Extract target values for training
    y_train = train_df[TARGET_COLUMN]
    
    # Separate input testing features by dropping target columns
    X_test = test_df.drop(columns=cols_to_drop)
    
    # Extract target values for testing
    y_test = test_df[TARGET_COLUMN]
    
    # Return separated training and testing feature matrices and targets
    return X_train, X_test, y_train, y_test


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculates R-squared, RMSE, MAE, and MAPE between true and predicted power values.
    """
    # Calculate R-squared coefficient of determination
    r2 = r2_score(y_true, y_pred)
    
    # Calculate Root Mean Squared Error (RMSE) in kilowatts
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calculate Mean Absolute Error (MAE) in kilowatts
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calculate Mean Absolute Percentage Error (MAPE) as a percentage
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100.0
    
    # Return computed metrics in a clean dictionary structure
    return {
        "R2": float(round(r2, 4)),
        "RMSE_kW": float(round(rmse, 4)),
        "MAE_kW": float(round(mae, 4)),
        "MAPE_percent": float(round(mape, 2))
    }


def optimize_and_train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    perform_cv: bool = True
) -> dict:
    """
    Trains and tunes Linear Regression, Decision Tree, and Random Forest models.
    Performs 5-fold cross-validation and hyperparameter tuning for tree-based models.
    """
    # Dictionary container to store fitted model objects and training diagnostics
    trained_artifacts = {}
    
    # Configure 5-fold cross-validation generator with random shuffling for reproducibility
    cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # -------------------------------------------------------------------------
    # 1. LINEAR REGRESSION (Baseline Parametric Model)
    # -------------------------------------------------------------------------
    # Log commencement of Linear Regression training
    print("\n" + "-" * 60)
    print(">>> [1/3] Training Linear Regression (Ordinary Least Squares)...")
    print("-" * 60)
    
    # Instantiate Scikit-Learn Ordinary Least Squares Linear Regression model
    linear_model = LinearRegression(fit_intercept=True)
    
    # Record starting timestamp for Linear Regression fitting
    lr_start_time = time.time()
    
    # Fit Linear Regression on full training dataset
    linear_model.fit(X_train, y_train)
    
    # Calculate total duration taken to fit Linear Regression
    lr_fit_time = time.time() - lr_start_time
    
    # Evaluate 5-fold cross-validation R2 scores across training folds
    lr_cv_scores = cross_val_score(linear_model, X_train, y_train, cv=cv_strategy, scoring="r2")
    
    # Predict training dataset power values using fitted Linear Regression
    lr_train_preds = linear_model.predict(X_train)
    
    # Compute performance metrics on training data for Linear Regression
    lr_train_metrics = compute_metrics(y_train, lr_train_preds)
    
    # Store Linear Regression artifacts and performance summary
    trained_artifacts["Linear Regression"] = {
        "model": linear_model,
        "fit_time_seconds": round(lr_fit_time, 4),
        "cv_r2_mean": float(round(lr_cv_scores.mean(), 4)),
        "cv_r2_std": float(round(lr_cv_scores.std(), 4)),
        "train_metrics": lr_train_metrics,
        "model_type": "Linear / Parametric"
    }
    
    # Print Linear Regression training performance summary
    print(f"Linear Regression Fit Time : {lr_fit_time:.3f} s")
    print(f"5-Fold CV R² Score         : {lr_cv_scores.mean():.4f} (±{lr_cv_scores.std():.4f})")
    print(f"Train R²: {lr_train_metrics['R2']:.4f} | RMSE: {lr_train_metrics['RMSE_kW']:.3f} kW | MAE: {lr_train_metrics['MAE_kW']:.3f} kW | MAPE: {lr_train_metrics['MAPE_percent']:.2f}%")
    
    # -------------------------------------------------------------------------
    # 2. DECISION TREE REGRESSOR (Non-Linear Single Tree Model)
    # -------------------------------------------------------------------------
    # Log commencement of Decision Tree optimization and training
    print("\n" + "-" * 60)
    print(">>> [2/3] Tuning and Training Decision Tree Regressor...")
    print("-" * 60)
    
    # Define hyperparameter grid for Decision Tree optimization
    dt_param_grid = {
        "max_depth": [8, 12, 16, 20],               # Maximum depth of tree to prevent overfitting
        "min_samples_split": [2, 5, 10],            # Minimum samples required to split an internal node
        "min_samples_leaf": [1, 2, 4]               # Minimum samples required at leaf node
    }
    
    # Instantiate base Decision Tree Regressor with fixed random seed
    base_dt = DecisionTreeRegressor(random_state=42)
    
    # Set up GridSearchCV to find optimal hyperparameters using 5-fold cross-validation
    dt_grid_search = GridSearchCV(
        estimator=base_dt,
        param_grid=dt_param_grid,
        cv=cv_strategy,
        scoring="r2",
        n_jobs=-1                                   # Utilize all CPU cores for parallel search
    )
    
    # Record starting timestamp for Decision Tree hyperparameter tuning
    dt_start_time = time.time()
    
    # Execute grid search on training features and targets
    dt_grid_search.fit(X_train, y_train)
    
    # Calculate elapsed tuning and training duration
    dt_fit_time = time.time() - dt_start_time
    
    # Retrieve the best tuned Decision Tree estimator from grid search
    best_dt_model = dt_grid_search.best_estimator_
    
    # Predict training set using the best Decision Tree model
    dt_train_preds = best_dt_model.predict(X_train)
    
    # Compute training metrics for the optimized Decision Tree
    dt_train_metrics = compute_metrics(y_train, dt_train_preds)
    
    # Store Decision Tree artifacts and results
    trained_artifacts["Decision Tree"] = {
        "model": best_dt_model,
        "best_params": dt_grid_search.best_params_,
        "fit_time_seconds": round(dt_fit_time, 4),
        "cv_r2_mean": float(round(dt_grid_search.best_score_, 4)),
        "train_metrics": dt_train_metrics,
        "model_type": "Non-Linear Decision Tree"
    }
    
    # Print Decision Tree training performance summary
    print(f"Decision Tree Tuning/Fit Time : {dt_fit_time:.3f} s")
    print(f"Optimal Hyperparameters       : {dt_grid_search.best_params_}")
    print(f"Best 5-Fold CV R² Score       : {dt_grid_search.best_score_:.4f}")
    print(f"Train R²: {dt_train_metrics['R2']:.4f} | RMSE: {dt_train_metrics['RMSE_kW']:.3f} kW | MAE: {dt_train_metrics['MAE_kW']:.3f} kW | MAPE: {dt_train_metrics['MAPE_percent']:.2f}%")
    
    # -------------------------------------------------------------------------
    # 3. RANDOM FOREST REGRESSOR (Ensemble Bagging Model)
    # -------------------------------------------------------------------------
    # Log commencement of Random Forest training
    print("\n" + "-" * 60)
    print(">>> [3/3] Training and Tuning Random Forest Regressor...")
    print("-" * 60)
    
    # Instantiate Random Forest Regressor configured with robust parameters
    rf_model = RandomForestRegressor(
        n_estimators=150,                           # Number of trees in the forest
        max_depth=16,                               # Depth ceiling to prevent individual tree overfit
        min_samples_split=4,                        # Minimum samples needed to split a node
        min_samples_leaf=2,                         # Minimum samples per leaf node
        max_features="sqrt",                        # Number of features to consider at each split
        random_state=42,                            # Fixed seed for reproducibility
        n_jobs=-1                                   # Parallelize tree construction across all CPU cores
    )
    
    # Record starting timestamp for Random Forest fitting
    rf_start_time = time.time()
    
    # Fit Random Forest ensemble on the training data
    rf_model.fit(X_train, y_train)
    
    # Calculate total fitting duration for Random Forest
    rf_fit_time = time.time() - rf_start_time
    
    # Evaluate 5-fold cross validation R2 for Random Forest
    rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=cv_strategy, scoring="r2", n_jobs=-1)
    
    # Predict training dataset power values using fitted Random Forest
    rf_train_preds = rf_model.predict(X_train)
    
    # Compute training metrics for Random Forest
    rf_train_metrics = compute_metrics(y_train, rf_train_preds)
    
    # Store Random Forest artifacts and results
    trained_artifacts["Random Forest"] = {
        "model": rf_model,
        "fit_time_seconds": round(rf_fit_time, 4),
        "cv_r2_mean": float(round(rf_cv_scores.mean(), 4)),
        "cv_r2_std": float(round(rf_cv_scores.std(), 4)),
        "train_metrics": rf_train_metrics,
        "model_type": "Ensemble Bagging"
    }
    
    # Print Random Forest training performance summary
    print(f"Random Forest Fit Time     : {rf_fit_time:.3f} s")
    print(f"5-Fold CV R² Score         : {rf_cv_scores.mean():.4f} (±{rf_cv_scores.std():.4f})")
    print(f"Train R²: {rf_train_metrics['R2']:.4f} | RMSE: {rf_train_metrics['RMSE_kW']:.3f} kW | MAE: {rf_train_metrics['MAE_kW']:.3f} kW | MAPE: {rf_train_metrics['MAPE_percent']:.2f}%")
    
    # Return all fitted model artifacts and training diagnostic metrics
    return trained_artifacts


def save_trained_models(
    trained_artifacts: dict,
    models_dir: str = "models"
) -> str:
    """
    Serializes all three trained models to disk as joblib pickle files,
    determines the champion model based on cross-validation R-squared,
    and updates the models/model_leaderboard.json metadata file.
    """
    # Build the full filesystem path for the models storage directory
    full_models_dir = os.path.join(PROJECT_ROOT_DIR, models_dir)
    
    # Ensure the models directory exists on the filesystem
    os.makedirs(full_models_dir, exist_ok=True)
    
    # Define standardized filename mapping for each algorithm
    filename_map = {
        "Linear Regression": "linear_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "Random Forest": "random_forest.pkl"
    }
    
    # Variable to keep track of the highest CV R-squared score
    best_cv_r2 = -float("inf")
    
    # Variable to store the name of the top-performing champion model
    champion_name = ""
    
    # Variable to store the champion model object
    champion_model_obj = None
    
    # List to collect structured records for leaderboard output
    leaderboard_records = []
    
    # Iterate through all trained models and their artifacts
    for model_name, info in trained_artifacts.items():
        # Retrieve the underlying Scikit-Learn model object
        model_obj = info["model"]
        
        # Get target filename from map
        target_filename = filename_map[model_name]
        
        # Build destination path for saving this model
        target_path = os.path.join(full_models_dir, target_filename)
        
        # Serialize and save the model object to disk
        joblib.dump(model_obj, target_path)
        
        # Print confirmation message for saved model
        print(f"[SAVED] {model_name} model saved to: {target_path}")
        
        # Extract CV R2 score
        cv_r2 = info["cv_r2_mean"]
        
        # Build a clean leaderboard dictionary record for this model
        record = {
            "Model": model_name,
            "R2": cv_r2,
            "Model_Type": info["model_type"],
            "Fit_Time_s": info["fit_time_seconds"],
            "CV_R2_5Fold": cv_r2,
            "Train_R2": info["train_metrics"]["R2"],
            "Train_RMSE_kW": info["train_metrics"]["RMSE_kW"],
            "Train_MAE_kW": info["train_metrics"]["MAE_kW"],
            "Train_MAPE_percent": info["train_metrics"]["MAPE_percent"]
        }
        
        # Append record to leaderboard list
        leaderboard_records.append(record)
        
        # Check if this model exceeds current best cross-validation performance
        if cv_r2 > best_cv_r2:
            # Update best CV R2 score
            best_cv_r2 = cv_r2
            
            # Update champion model name
            champion_name = model_name
            
            # Update champion model object
            champion_model_obj = model_obj
            
    # Define destination path for the champion model (best_model.pkl)
    champion_save_path = os.path.join(full_models_dir, "best_model.pkl")
    
    # Serialize and save the champion model as best_model.pkl
    joblib.dump(champion_model_obj, champion_save_path)
    
    # Print status message indicating champion model persistence
    print(f"[CHAMPION] Top model '{champion_name}' (CV R² = {best_cv_r2:.4f}) saved to: {champion_save_path}")
    
    # Sort leaderboard records by CV R2 descending
    leaderboard_records.sort(key=lambda x: x["CV_R2_5Fold"], reverse=True)
    
    # Construct complete leaderboard metadata dictionary
    leaderboard_payload = {
        "champion_model": champion_name,
        "champion_cv_r2": best_cv_r2,
        "models_benchmarked": list(trained_artifacts.keys()),
        "leaderboard": leaderboard_records
    }
    
    # Build file path for model leaderboard JSON
    leaderboard_json_path = os.path.join(full_models_dir, "model_leaderboard.json")
    
    # Open leaderboard JSON file in write mode
    with open(leaderboard_json_path, "w") as f:
        # Dump formatted JSON with 4-space indentation
        json.dump(leaderboard_payload, f, indent=4)
        
    # Print confirmation that leaderboard metadata was saved
    print(f"[SAVED] Training leaderboard summary saved to: {leaderboard_json_path}")
    
    # Return the name of the champion model
    return champion_name


def main():
    """
    Main orchestration routine for training Linear Regression, Decision Tree, and Random Forest.
    """
    # Print banner header
    print("=" * 70)
    print("      CNC MACHINING POWER PREDICTION - TRAINING PIPELINE (src/train.py)")
    print("      Algorithms: Linear Regression | Decision Tree | Random Forest")
    print("=" * 70)
    
    # Load or prepare preprocessed training and testing datasets
    X_train, X_test, y_train, y_test = load_training_and_testing_data()
    
    # Print dataset dimensions
    print(f"Training Sample Count : {X_train.shape[0]:,}")
    print(f"Feature Column Count  : {X_train.shape[1]}")
    
    # Train and optimize all three target algorithms
    trained_artifacts = optimize_and_train_models(X_train, y_train, perform_cv=True)
    
    # Save all fitted models and serialize leaderboard results
    champion_name = save_trained_models(trained_artifacts, models_dir="models")
    
    # Print concluding banner
    print("\n" + "=" * 70)
    print(f"   TRAINING COMPLETED SUCCESSFULLY. CHAMPION ALGORITHM: {champion_name}")
    print("=" * 70)


# Execute main() function when invoked directly from the terminal
if __name__ == "__main__":
    # Call the main entry point
    main()
