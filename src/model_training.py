"""
Model Training and Benchmarking Module
Trains multiple ML algorithms (XGBoost, Random Forest, HistGradientBoosting, Ridge, MLP Regressor),
computes comprehensive performance metrics, benchmarks their accuracy/speed,
and serializes the champion model for inference.
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

TARGET_COLUMN = "Power_Consumption_kW"


def load_processed_data(processed_dir: str = "data/processed"):
    """
    Loads preprocessed training and testing data splits.
    """
    train_path = os.path.join(processed_dir, "train_features.csv")
    test_path = os.path.join(processed_dir, "test_features.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Processed data files not found. Run data_preprocessing.py first.")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]
    
    return X_train, X_test, y_train, y_test


def get_candidate_models():
    """
    Defines diverse regression models for benchmarking.
    """
    models = {
        "XGBoost Regressor": xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.07,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=16,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.08,
            max_depth=8,
            random_state=42
        ),
        "Ridge Regression": Ridge(
            alpha=1.5,
            random_state=42
        ),
        "Neural Network (MLP)": MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=250,
            early_stopping=True,
            random_state=42
        )
    }
    return models


def evaluate_predictions(y_true, y_pred):
    """
    Computes key regression metrics.
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100.0
    return {
        "R2": float(r2),
        "RMSE_kW": float(rmse),
        "MAE_kW": float(mae),
        "MAPE_percent": float(mape)
    }


def train_and_benchmark(models_dir: str = "models"):
    """
    Executes model training, evaluation, comparison leaderboard creation,
    and champion model export.
    """
    os.makedirs(models_dir, exist_ok=True)
    X_train, X_test, y_train, y_test = load_processed_data()
    
    print(f"\n=======================================================")
    print(f"   MACHINING POWER PREDICTION - MODEL BENCHMARKING    ")
    print(f"=======================================================")
    print(f"Training samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}")
    print(f"Feature count:    {X_train.shape[1]}")
    
    candidate_models = get_candidate_models()
    leaderboard = []
    trained_models = {}
    test_predictions = {}
    
    best_model_name = None
    best_r2 = -float("inf")
    best_model_obj = None
    
    for name, model in candidate_models.items():
        print(f"\n---> Training [{name}]...")
        start_time = time.time()
        model.fit(X_train, y_train)
        train_duration = time.time() - start_time
        
        y_pred = model.predict(X_test)
        metrics = evaluate_predictions(y_test, y_pred)
        metrics["Model"] = name
        metrics["Train_Time_s"] = round(train_duration, 3)
        
        leaderboard.append(metrics)
        trained_models[name] = model
        test_predictions[name] = y_pred.tolist()
        
        print(f"     Finished in {train_duration:.2f}s | R2: {metrics['R2']:.4f} | RMSE: {metrics['RMSE_kW']:.3f} kW | MAE: {metrics['MAE_kW']:.3f} kW | MAPE: {metrics['MAPE_percent']:.2f}%")
        
        if metrics["R2"] > best_r2:
            best_r2 = metrics["R2"]
            best_model_name = name
            best_model_obj = model
            
    # Sort leaderboard by R2 descending
    leaderboard_df = pd.DataFrame(leaderboard).sort_values(by="R2", ascending=False).reset_index(drop=True)
    
    print("\n=======================================================")
    print("                 BENCHMARK LEADERBOARD                ")
    print("=======================================================")
    print(leaderboard_df.to_string(index=False))
    print("=======================================================")
    print(f"[BEST MODEL] Champion Model Selected: {best_model_name} (R2 = {best_r2:.4f})")
    
    # Save champion model
    best_model_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(best_model_obj, best_model_path)
    print(f"Saved champion model to: {best_model_path}")
    
    # Save individual models
    for m_name, m_obj in trained_models.items():
        clean_name = m_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        m_path = os.path.join(models_dir, f"{clean_name}.pkl")
        joblib.dump(m_obj, m_path)
    
    # Save leaderboard results
    leaderboard_path = os.path.join(models_dir, "model_leaderboard.json")
    with open(leaderboard_path, "w") as f:
        json.dump({
            "champion_model": best_model_name,
            "champion_r2": best_r2,
            "leaderboard": leaderboard
        }, f, indent=4)
        
    # Save test predictions for evaluation and plotting
    preds_path = os.path.join(models_dir, "test_predictions.json")
    with open(preds_path, "w") as f:
        json.dump({
            "y_test": y_test.tolist(),
            "predictions": test_predictions
        }, f)
        
    return best_model_name, leaderboard_df


if __name__ == "__main__":
    train_and_benchmark()
