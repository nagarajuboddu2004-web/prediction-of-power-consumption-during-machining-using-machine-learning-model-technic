"""
Model Testing, Diagnostic Evaluation, and Visualization Module.
Evaluates the three trained machine learning models on the unseen holdout test set:
  1. Linear Regression
  2. Decision Tree Regressor
  3. Random Forest Regressor
Generates quantitative leaderboard tables, residual statistics, and diagnostic figures.
Every single line of code is documented with an inline comment explaining its function.
"""

# Import operating system interfaces for directory management and file paths
import os

# Import system-specific parameters and system path manipulation
import sys

# Import time module to benchmark prediction latency on test samples
import time

# Import json module to serialize test metrics and evaluation summaries
import json

# Import joblib to deserialize trained Scikit-Learn models from disk
import joblib

# Import pathlib for clean cross-platform filesystem paths
from pathlib import Path

# Import numpy for numerical computations and array operations
import numpy as np

# Import pandas for structured tabular data manipulation and CSV handling
import pandas as pd

# Import matplotlib for figure plotting
import matplotlib.pyplot as plt

# Import seaborn for modern statistical data visualization styling
import seaborn as sns

# Import regression evaluation metrics from Scikit-Learn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, max_error

# Determine the absolute path of this script
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent

# Determine the project root directory
PROJECT_ROOT_DIR = CURRENT_SCRIPT_DIR.parent

# Ensure project root directory is available in the Python system path
if str(PROJECT_ROOT_DIR) not in sys.path:
    # Insert project root into sys.path
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

# Define target column name
TARGET_COLUMN = "Power_Consumption_kW"


def load_test_dataset(processed_dir: str = "data/processed"):
    """
    Loads preprocessed test features and true target values.
    """
    # Build the full file path for the test features CSV
    test_csv_path = os.path.join(PROJECT_ROOT_DIR, processed_dir, "test_features.csv")
    
    # Check if the test features CSV exists on disk
    if not os.path.exists(test_csv_path):
        # Raise an exception if the file cannot be located
        raise FileNotFoundError(f"Test data file not found at: {test_csv_path}. Run preprocessing first.")
        
    # Read the test features CSV into a pandas DataFrame
    test_df = pd.read_csv(test_csv_path)
    
    # Separate input feature columns by dropping target columns to avoid leakage
    target_cols = [TARGET_COLUMN, "Carbon_Emission_Rate_kgCO2e_hr", "Specific_Carbon_Emission_gCO2e_cm3"]
    cols_to_drop = [c for c in target_cols if c in test_df.columns]
    X_test = test_df.drop(columns=cols_to_drop)
    
    # Extract the target column values as the ground truth vector
    y_test = test_df[TARGET_COLUMN]
    
    # Return feature matrix X_test and ground truth target y_test
    return X_test, y_test


def load_all_models(models_dir: str = "models") -> dict:
    """
    Loads Linear Regression, Decision Tree, and Random Forest models from the models directory.
    """
    # Build full filesystem path to the models directory
    full_models_dir = os.path.join(PROJECT_ROOT_DIR, models_dir)
    
    # Dictionary mapping display names to corresponding pickle filenames
    models_to_load = {
        "Linear Regression": "linear_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "Random Forest": "random_forest.pkl"
    }
    
    # Dictionary to hold loaded model objects
    loaded_models = {}
    
    # Iterate through the model definitions
    for name, filename in models_to_load.items():
        # Build the full path to the model pickle file
        model_file_path = os.path.join(full_models_dir, filename)
        
        # Check if the model pickle file exists
        if os.path.exists(model_file_path):
            # Load the pickled model object from disk
            loaded_models[name] = joblib.load(model_file_path)
            # Log successful model loading
            print(f"[LOADED] Successfully loaded '{name}' from {filename}")
        else:
            # Print warning if model file is missing
            print(f"[WARNING] Model file not found for '{name}' at {model_file_path}")
            
    # Check if any models were successfully loaded
    if not loaded_models:
        # Raise error if no models could be loaded
        raise FileNotFoundError("No trained models found in the models directory. Run train.py first.")
        
    # Return dictionary containing all loaded models
    return loaded_models


def evaluate_model_on_test(model, X_test: pd.DataFrame, y_test: pd.Series) -> tuple:
    """
    Computes predictions, evaluation metrics, and latency for a single model.
    """
    # Record starting timestamp to measure test prediction duration
    t_start = time.time()
    
    # Generate predictions for all test samples
    y_pred = model.predict(X_test)
    
    # Calculate elapsed inference time in seconds
    inference_duration = time.time() - t_start
    
    # Calculate latency in milliseconds per 1,000 samples
    latency_ms_per_1k = (inference_duration / len(X_test)) * 1000.0 * 1000.0
    
    # Calculate R-squared coefficient of determination on unseen test data
    r2 = r2_score(y_test, y_pred)
    
    # Calculate Root Mean Squared Error (RMSE) in kilowatts
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Calculate Mean Absolute Error (MAE) in kilowatts
    mae = mean_absolute_error(y_test, y_pred)
    
    # Calculate Mean Absolute Percentage Error (MAPE) as percentage
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100.0
    
    # Calculate Maximum Residual Absolute Error in kilowatts
    max_err = max_error(y_test, y_pred)
    
    # Calculate residual errors (true minus predicted)
    residuals = y_test - y_pred
    
    # Structure metrics in a dictionary
    metrics = {
        "R2": float(round(r2, 4)),
        "RMSE_kW": float(round(rmse, 4)),
        "MAE_kW": float(round(mae, 4)),
        "MAPE_percent": float(round(mape, 2)),
        "Max_Error_kW": float(round(max_err, 4)),
        "Latency_ms_per_1k": float(round(latency_ms_per_1k, 2))
    }
    
    # Return metrics dictionary, predictions array, and residuals series
    return metrics, y_pred, residuals


def generate_diagnostic_plots(
    results: dict,
    y_test: pd.Series,
    figures_dir: str = "reports/figures"
):
    """
    Creates publication-quality comparison charts:
      1. Model Comparison Bar Chart (R², RMSE, MAE)
      2. Actual vs. Predicted Parity Scatter Plots
      3. Residual Error Distribution Histograms
    """
    # Build full filesystem path to figures directory
    full_figures_dir = os.path.join(PROJECT_ROOT_DIR, figures_dir)
    
    # Ensure figures output directory exists
    os.makedirs(full_figures_dir, exist_ok=True)
    
    # Set modern Seaborn aesthetic style with white grid background
    sns.set_theme(style="whitegrid", palette="muted")
    
    # Extract model names
    model_names = list(results.keys())
    
    # -------------------------------------------------------------------------
    # 1. Model Comparison Bar Chart (R2, RMSE, MAE)
    # -------------------------------------------------------------------------
    # Create matplotlib figure with 1 row and 3 columns of subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    
    # Define custom color palette for the 3 algorithms
    colors = ["#4A5568", "#2B6CB0", "#2F855A"]
    
    # Subplot 1: R2 Score
    r2_values = [results[m]["metrics"]["R2"] for m in model_names]
    axes[0].bar(model_names, r2_values, color=colors, edgecolor="black", alpha=0.85)
    axes[0].set_title("Test R² Score (Higher is Better)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("R² Coefficient of Determination", fontsize=11)
    axes[0].set_ylim(0, 1.05)
    for i, v in enumerate(r2_values):
        axes[0].text(i, v + 0.02, f"{v:.4f}", ha="center", fontweight="bold", fontsize=10)
    axes[0].tick_params(axis="x", rotation=15)
    
    # Subplot 2: RMSE in kW
    rmse_values = [results[m]["metrics"]["RMSE_kW"] for m in model_names]
    axes[1].bar(model_names, rmse_values, color=colors, edgecolor="black", alpha=0.85)
    axes[1].set_title("Root Mean Squared Error (Lower is Better)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("RMSE (kW)", fontsize=11)
    for i, v in enumerate(rmse_values):
        axes[1].text(i, v + 0.05, f"{v:.3f} kW", ha="center", fontweight="bold", fontsize=10)
    axes[1].tick_params(axis="x", rotation=15)
    
    # Subplot 3: MAE in kW
    mae_values = [results[m]["metrics"]["MAE_kW"] for m in model_names]
    axes[2].bar(model_names, mae_values, color=colors, edgecolor="black", alpha=0.85)
    axes[2].set_title("Mean Absolute Error (Lower is Better)", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("MAE (kW)", fontsize=11)
    for i, v in enumerate(mae_values):
        axes[2].text(i, v + 0.03, f"{v:.3f} kW", ha="center", fontweight="bold", fontsize=10)
    axes[2].tick_params(axis="x", rotation=15)
    
    # Adjust layout padding
    plt.tight_layout()
    
    # Save comparison figure to disk
    comp_plot_path = os.path.join(full_figures_dir, "model_comparison_bar.png")
    fig.savefig(comp_plot_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Model comparison bar chart saved to: {comp_plot_path}")
    
    # -------------------------------------------------------------------------
    # 2. Actual vs. Predicted Parity Scatter Plots
    # -------------------------------------------------------------------------
    # Create matplotlib figure with 1 row and 3 columns for parity plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    
    # Iterate through each model and plot scatter vs parity line
    for idx, name in enumerate(model_names):
        ax = axes[idx]
        y_pred = results[name]["predictions"]
        r2_val = results[name]["metrics"]["R2"]
        rmse_val = results[name]["metrics"]["RMSE_kW"]
        
        # Plot scatter points with transparency
        ax.scatter(y_test, y_pred, alpha=0.35, s=18, color=colors[idx], edgecolors="none")
        
        # Plot perfect agreement parity line (y = x)
        min_val = min(y_test.min(), np.min(y_pred)) - 1
        max_val = max(y_test.max(), np.max(y_pred)) + 1
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.8, label="Ideal 1:1 Parity")
        
        ax.set_title(f"{name}\nR² = {r2_val:.4f} | RMSE = {rmse_val:.3f} kW", fontsize=11, fontweight="bold")
        ax.set_xlabel("Actual Power (kW)", fontsize=10)
        ax.set_ylabel("Predicted Power (kW)", fontsize=10)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.legend(loc="upper left")
        
    plt.tight_layout()
    parity_plot_path = os.path.join(full_figures_dir, "actual_vs_predicted.png")
    fig.savefig(parity_plot_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Actual vs. Predicted parity plot saved to: {parity_plot_path}")
    
    # -------------------------------------------------------------------------
    # 3. Residual Error Distribution Histograms
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    
    for idx, name in enumerate(model_names):
        ax = axes[idx]
        residuals = results[name]["residuals"]
        
        # Plot KDE and histogram of residuals
        sns.histplot(residuals, kde=True, ax=ax, color=colors[idx], bins=35, stat="density", edgecolor="black", alpha=0.6)
        
        # Plot vertical zero error line
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error")
        
        mae_val = results[name]["metrics"]["MAE_kW"]
        ax.set_title(f"{name} Residuals\nMean: {residuals.mean():.3f} kW | MAE: {mae_val:.3f} kW", fontsize=11, fontweight="bold")
        ax.set_xlabel("Residual Error (Actual - Predicted) [kW]", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(loc="upper right")
        
    plt.tight_layout()
    resid_plot_path = os.path.join(full_figures_dir, "residual_distribution.png")
    fig.savefig(resid_plot_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Residual distribution plot saved to: {resid_plot_path}")

    # -------------------------------------------------------------------------
    # 4. Carbon Emissions & Green Machining Diagnostic Plot
    # -------------------------------------------------------------------------
    raw_test_path = os.path.join(PROJECT_ROOT_DIR, "data", "raw", "machining_power_consumption_12k.csv")
    if os.path.exists(raw_test_path):
        df_raw = pd.read_csv(raw_test_path)
        fig_carbon, axes_c = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
        
        # Subplot 1: Carbon Emission vs. Power (Linear scaling across regional grids)
        p_sample = np.linspace(2, 25, 50)
        axes_c[0].plot(p_sample, p_sample * 0.475, label="Global Avg (0.475)", color="#2563EB", linewidth=2)
        axes_c[0].plot(p_sample, p_sample * 0.708, label="India Grid (0.708)", color="#DC2626", linewidth=2)
        axes_c[0].plot(p_sample, p_sample * 0.385, label="US Grid (0.385)", color="#D97706", linewidth=2)
        axes_c[0].plot(p_sample, p_sample * 0.255, label="EU Grid (0.255)", color="#059669", linewidth=2)
        axes_c[0].plot(p_sample, p_sample * 0.045, label="Renewable (0.045)", color="#10B981", linestyle="--", linewidth=2)
        axes_c[0].set_title("Operational Carbon vs Electrical Power Demand", fontsize=11, fontweight="bold")
        axes_c[0].set_xlabel("Predicted Power (kW)", fontsize=10)
        axes_c[0].set_ylabel("Carbon Rate (kg CO2e / hr)", fontsize=10)
        axes_c[0].legend(loc="upper left", fontsize=9)
        axes_c[0].grid(True, linestyle=":", alpha=0.6)
        
        # Subplot 2: Carbon Footprint by Coolant Strategy
        if "Carbon_Emission_Rate_kgCO2e_hr" in df_raw.columns:
            cool_ce = df_raw.groupby("Coolant_Condition")["Carbon_Emission_Rate_kgCO2e_hr"].mean().reset_index()
            axes_c[1].bar(cool_ce["Coolant_Condition"], cool_ce["Carbon_Emission_Rate_kgCO2e_hr"], color=["#10B981", "#3B82F6", "#F59E0B", "#EF4444"], edgecolor="black", alpha=0.85)
            axes_c[1].set_title("Mean Carbon Emission Rate by Coolant Mode", fontsize=11, fontweight="bold")
            axes_c[1].set_ylabel("Mean Carbon Rate (kg CO2e / hr)", fontsize=10)
            axes_c[1].tick_params(axis="x", rotation=15)
            for i, v in enumerate(cool_ce["Carbon_Emission_Rate_kgCO2e_hr"]):
                axes_c[1].text(i, v + 0.05, f"{v:.2f}", ha="center", fontweight="bold", fontsize=9)
                
        # Subplot 3: Specific Carbon Emission (SCE) by Material
        if "Specific_Carbon_Emission_gCO2e_cm3" in df_raw.columns:
            mat_sce = df_raw.groupby("Workpiece_Material")["Specific_Carbon_Emission_gCO2e_cm3"].median().reset_index().sort_values(by="Specific_Carbon_Emission_gCO2e_cm3")
            axes_c[2].barh(mat_sce["Workpiece_Material"], mat_sce["Specific_Carbon_Emission_gCO2e_cm3"], color="#6366F1", edgecolor="black", alpha=0.85)
            axes_c[2].set_title("Median Specific Carbon Emission by Alloy", fontsize=11, fontweight="bold")
            axes_c[2].set_xlabel("Specific Carbon (g CO2e / cm3)", fontsize=10)
            for i, v in enumerate(mat_sce["Specific_Carbon_Emission_gCO2e_cm3"]):
                axes_c[2].text(v + 0.05, i, f" {v:.2f}", va="center", fontweight="bold", fontsize=9)
                
        plt.tight_layout()
        carbon_plot_path = os.path.join(full_figures_dir, "carbon_emission_analysis.png")
        fig_carbon.savefig(carbon_plot_path, bbox_inches="tight")
        plt.close(fig_carbon)
        print(f"[PLOT] Carbon emission analysis plot saved to: {carbon_plot_path}")


def main():
    """
    Main orchestration routine for testing, benchmarking, and generating diagnostic plots.
    """
    # Print banner header
    print("=" * 70)
    print("      CNC MACHINING POWER PREDICTION - TESTING & EVALUATION (src/test.py)")
    print("      Evaluating: Linear Regression | Decision Tree | Random Forest")
    print("=" * 70)
    
    # Load test dataset
    X_test, y_test = load_test_dataset()
    print(f"Unseen Holdout Test Samples : {len(X_test):,}")
    print(f"Feature Dimension Count     : {X_test.shape[1]}")
    
    # Load all 3 trained models
    loaded_models = load_all_models()
    
    # Dictionary to store all evaluation results
    results = {}
    leaderboard_rows = []
    
    # Iterate and evaluate each model on unseen test data
    for name, model in loaded_models.items():
        print(f"\n--- Testing [{name}] on {len(X_test):,} samples...")
        metrics, preds, residuals = evaluate_model_on_test(model, X_test, y_test)
        
        # Store results
        results[name] = {
            "metrics": metrics,
            "predictions": preds,
            "residuals": residuals
        }
        
        # Create summary row for leaderboard table
        row = {"Model": name, **metrics}
        leaderboard_rows.append(row)
        
        print(f"    R² Score : {metrics['R2']:.4f}")
        print(f"    RMSE     : {metrics['RMSE_kW']:.3f} kW")
        print(f"    MAE      : {metrics['MAE_kW']:.3f} kW")
        print(f"    MAPE     : {metrics['MAPE_percent']:.2f}%")
        print(f"    Max Error: {metrics['Max_Error_kW']:.3f} kW")
        print(f"    Latency  : {metrics['Latency_ms_per_1k']:.2f} ms / 1k samples")
        
    # Build DataFrame from leaderboard rows and sort by R2 descending
    leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values(by="R2", ascending=False).reset_index(drop=True)
    
    # Print formatted leaderboard
    print("\n" + "=" * 70)
    print("              UNSEEN TEST SET BENCHMARK LEADERBOARD")
    print("=" * 70)
    print(leaderboard_df.to_string(index=False))
    print("=" * 70)
    
    # Identify the best performing model on the test set
    best_test_model = leaderboard_df.iloc[0]["Model"]
    best_test_r2 = leaderboard_df.iloc[0]["R2"]
    print(f"\n[WINNER] Highest Test Accuracy: {best_test_model} (R² = {best_test_r2:.4f})")
    
    # Save test metrics summary JSON
    reports_dir = os.path.join(PROJECT_ROOT_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    test_metrics_path = os.path.join(reports_dir, "test_metrics.json")
    
    # Structure json output payload
    output_payload = {
        "evaluated_models": list(loaded_models.keys()),
        "champion_model_on_test": best_test_model,
        "leaderboard": leaderboard_df.to_dict(orient="records")
    }
    
    # Write test metrics to JSON
    with open(test_metrics_path, "w") as f:
        json.dump(output_payload, f, indent=4)
    print(f"[SAVED] Test evaluation metrics saved to: {test_metrics_path}")
    
    # Generate and save publication-quality diagnostic plots
    print("\nGenerating diagnostic evaluation plots...")
    generate_diagnostic_plots(results, y_test, figures_dir="reports/figures")
    
    # Concluding banner
    print("\n" + "=" * 70)
    print("      TESTING AND DIAGNOSTIC EVALUATION COMPLETE SUCCESSFULLY")
    print("=" * 70)


# Execute main function when script is run directly
if __name__ == "__main__":
    main()
