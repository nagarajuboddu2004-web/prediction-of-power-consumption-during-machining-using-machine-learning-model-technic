# Machining Power Consumption Prediction - Evaluation Summary

## Executive Summary
This evaluation reviews model accuracy, residual behavior, and physical parameter significance for predicting CNC machining electrical power consumption ($P_{total}$ in kW).
The models were trained on 10,000 observations and evaluated on 2,500 held-out test observations.

## Champion Model
- **Algorithm**: `Neural Network (MLP)`
- **R² Score**: `0.9872`
- **Root Mean Squared Error (RMSE)**: `0.461 kW`
- **Mean Absolute Error (MAE)**: `0.321 kW`
- **Mean Absolute Percentage Error (MAPE)**: `4.60%`

## Algorithm Benchmark Comparison
| Model | R² Score | RMSE (kW) | MAE (kW) | MAPE (%) | Train Time (s) |
|---|---|---|---|---|---|
| **Neural Network (MLP)** | 0.9872 | 0.461 | 0.321 | 4.60% | 4.80 |
| **HistGradientBoosting** | 0.9792 | 0.588 | 0.395 | 5.46% | 2.16 |
| **XGBoost Regressor** | 0.9787 | 0.594 | 0.395 | 5.36% | 0.96 |
| **Random Forest** | 0.9623 | 0.791 | 0.523 | 7.28% | 3.16 |
| **Ridge Regression** | 0.8391 | 1.634 | 1.144 | 17.55% | 0.01 |

## Key Technical Insights & Physics Interpretation
1. **Dominant Power Drivers**: Material Removal Rate ($MRR$), cutting speed ($v_c$), and depth of cut ($a_p$) constitute the highest feature importances, conforming directly with Kienzle's empirical cutting power formulations.
2. **Workpiece Material Dependency**: Difficult-to-cut superalloys (Inconel 718 and Ti-6Al-4V) dramatically increase active power compared to aluminum alloys (Al 6061-T6) due to higher shear strength and specific cutting energy.
3. **Tool Wear Impact**: Progressing tool flank wear ($VB$) produces a measurable upward drift in required electrical power due to increased rubbing friction and contact land force.
4. **Generalization Quality**: Residual distributions are centered tightly around zero without systematic bias or heteroscedastic fan-out, proving strong generalization for real-world CNC process optimization.
