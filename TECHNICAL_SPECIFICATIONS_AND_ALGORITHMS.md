# 📘 CNC Machining Power Consumption Prediction: Technical Architecture, Algorithms & Technology Stack Specification

---

## Executive Overview

This document provides a comprehensive technical reference for the **CNC Machining Power Consumption Prediction System**. The system is an end-to-end, physics-grounded Machine Learning framework designed for smart manufacturing, digital twins, green machining optimization, and tool condition monitoring. It predicts instantaneous active electrical power consumption ($P_{\text{total}}$ in kilowatts) during industrial CNC milling and turning operations across multiple aerospace and industrial alloys.

---

## 1. Complete Technology Stack

The project leverages a modern, robust, and performant Python ecosystem engineered for reproducible data science, high-accuracy regression modeling, and responsive web deployment.

```
+-------------------------------------------------------------------------------+
|                             PRESENTATION LAYER                                |
|   Streamlit (v1.28+) Web Dashboard  |  Interactive Sliders  |  Power Gauge    |
|   Energy Cost Calculator ($/hr)     |  What-If Curves       |  CSV Exporter   |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|                              INFERENCE ENGINE                                 |
|   CLI Entry (predict.py)  |  Fast Vectorized Array Ingestion (predict_power)   |
|   Automated Kinematic Imputation (RPM, Feed Speed, MRR derivations)           |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|                           MACHINE LEARNING MODELS                             |
|  * Multi-Layer Perceptron (MLPRegressor) [Champion: R²=0.9872]                |
|  * HistGradientBoostingRegressor         [R²=0.9792]                          |
|  * XGBoost Regressor (XGBRegressor)      [R²=0.9787]                          |
|  * Random Forest Regressor               [R²=0.9623]                          |
|  * Ridge Regularized Linear Regression   [R²=0.8391]                          |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|                         FEATURE PREPROCESSING PIPELINE                        |
|   Domain Feature Engineering (Speed×Feed, Depth Ratio, MRR/Flute, Wear Ratio) |
|   ColumnTransformer: StandardScaler (17 Features) + OneHotEncoder (15 Dummies)|
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|                       PHYSICS-BASED SYNTHESIS ENGINE                          |
|   Kienzle Tangential Force Model | Flank Wear Degradation | Coolant/Coating   |
|   Multi-Subsystem Power Breakdown (Base + Spindle + Feed + Coolant + Cut)     |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|                        STORAGE & SERIALIZATION LAYER                          |
|   Raw Data (CSV) | Processed Sets (CSV) | Pickled Models (.pkl) | JSON Meta   |
+-------------------------------------------------------------------------------+
```

### 1.1 Technology Breakdown

| Component Category | Technology / Library | Minimum Version | Exact Purpose in Codebase |
|---|---|---|---|
| **Core Runtime** | **Python** | `>=3.10` | Base programming language environment |
| **Deep Learning / Neural Net** | **Scikit-Learn (`MLPRegressor`)** | `>=1.3.0` | Champion feedforward neural network model |
| **Gradient Boosting** | **XGBoost (`xgb.XGBRegressor`)** | `>=2.0.0` | Extreme gradient boosting tree ensemble |
| **Ensemble Learning** | **Scikit-Learn (`HistGradientBoosting`, `RandomForest`)** | `>=1.3.0` | Histogram-based and bagged ensemble models |
| **Linear Modeling** | **Scikit-Learn (`Ridge`)** | `>=1.3.0` | $L_2$-penalized linear baseline regression |
| **Data Transformations** | **Scikit-Learn (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`)** | `>=1.3.0` | Leakage-free preprocessing pipeline |
| **Numerical Computing** | **NumPy** | `>=1.24.0` | Array operations, kinematic math, trigonometry |
| **Data Manipulation** | **Pandas** | `>=2.0.0` | Dataframe handling, data generation, CSV storage |
| **Model Serialization** | **Joblib** | `>=1.3.0` | Exporting trained models and transformation pipelines |
| **Web Dashboard** | **Streamlit** | `>=1.28.0` | Interactive UI, live inference, sensitivity simulator |
| **Diagnostic Plotting** | **Matplotlib** & **Seaborn** | `>=3.7.0` / `>=0.12.0` | Parity plots, residual histograms, feature rankings |
| **Operating Environment** | **Windows / PowerShell / Batch** | OS Windows | Automation scripts (`run_app.bat`, `run_pipeline.py`) |

---

## 2. Comprehensive Breakdown of Machine Learning Algorithms

To determine the most accurate and computationally efficient model for CNC active power prediction, five diverse algorithm paradigms were implemented, trained on 10,000 observations, and evaluated on 2,500 held-out test records.

### 2.1 Summary Benchmark Leaderboard

| Model Rank | Algorithm Paradigm | $R^2$ Score | RMSE ($\text{kW}$) | MAE ($\text{kW}$) | MAPE ($\%$) | Training Time (s) | Status |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 **1** | **Multi-Layer Perceptron (MLP)** | **0.9872** | **0.461** | **0.321** | **4.60%** | 4.80 s | **Champion Model** |
| 🥈 **2** | **HistGradientBoosting** | **0.9792** | **0.588** | **0.395** | **5.46%** | 2.16 s | Strong Contender |
| 🥉 **3** | **XGBoost Regressor** | **0.9787** | **0.594** | **0.395** | **5.36%** | 0.96 s | Fastest Booster |
| **4** | **Random Forest Regressor** | **0.9623** | **0.791** | **0.523** | **7.28%** | 3.16 s | Solid Ensemble |
| **5** | **Ridge Regression ($L_2$)** | **0.8391** | **1.634** | **1.144** | **17.55%** | 0.01 s | Linear Baseline |

---

### 2.2 In-Depth Algorithmic Analysis

#### 1. Champion Model: Neural Network (Multi-Layer Perceptron - `MLPRegressor`)
* **Architecture**:
  * **Input Layer**: 32 transformed continuous features (StandardScaled numericals + OneHot encoded categoricals).
  * **Hidden Layer 1**: 128 neurons, fully connected with **ReLU** ($\text{ReLU}(z) = \max(0, z)$) activation.
  * **Hidden Layer 2**: 64 neurons, fully connected with **ReLU** activation.
  * **Output Layer**: 1 linear output neuron ($P_{\text{total}}$ in kW).
* **Optimization & Training Strategy**:
  * **Optimizer**: Adam (Adaptive Moment Estimation) stochastic gradient descent.
  * **Loss Function**: Mean Squared Error ($MSE = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$).
  * **Regularization**: $L_2$ penalty parameter $\alpha = 0.0001$.
  * **Early Stopping**: Activated with 10% validation holdout to prevent overfitting.
  * **Max Iterations**: 250 epochs.
* **Why it Won**: Machining power contains smooth, continuous non-linear interactions (e.g., $P_{\text{cut}} \propto v_c \cdot a_p \cdot h^{1-m_c}$, quadratic spindle drag $N^2$, and continuous wear curves). While tree-based models approximate continuous response surfaces using orthogonal step functions, the neural network models smooth continuous multi-dimensional curvature with superior fidelity, achieving the lowest test error ($MAE = 0.321\text{ kW}$, $MAPE = 4.60\%$).

#### 2. HistGradientBoostingRegressor
* **Architecture & Mechanics**:
  * Scikit-Learn's implementation inspired by LightGBM.
  * Bins continuous input features into discrete 256 integer bins (histograms), drastically accelerating split evaluations and memory efficiency.
* **Hyperparameters**:
  * `max_iter`: 250 boosting stages.
  * `learning_rate`: 0.08 shrinkage rate.
  * `max_depth`: 8 levels.
  * `random_state`: 42.
* **Characteristics**: Highly robust to non-linear interactions, handles binned representations without overfitting, achieved $R^2 = 0.9792$.

#### 3. XGBoost Regressor (`xgb.XGBRegressor`)
* **Architecture & Mechanics**:
  * Optimized distributed gradient boosting library implementing exact greedy and histogram-based tree splitting with second-order Taylor expansion of the loss function.
* **Hyperparameters**:
  * `n_estimators`: 300 gradient boosted trees.
  * `learning_rate`: 0.07.
  * `max_depth`: 6.
  * `subsample`: 0.85 (stochastic row subsampling).
  * `colsample_bytree`: 0.85 (column subsampling per tree).
  * `n_jobs`: -1 (parallel multi-core CPU execution).
* **Characteristics**: Exceptional training speed ($0.96\text{ s}$ for 300 trees on 10,000 rows) while yielding virtually identical accuracy to HistGBM ($R^2 = 0.9787$, $RMSE = 0.594\text{ kW}$).

#### 4. Random Forest Regressor (`RandomForestRegressor`)
* **Architecture & Mechanics**:
  * Bagging (Bootstrap Aggregating) ensemble of 150 decorrelated decision trees trained on random bootstrapped subsets with random feature subsets at each node.
* **Hyperparameters**:
  * `n_estimators`: 150 trees.
  * `max_depth`: 16 levels.
  * `min_samples_split`: 4 samples.
  * `random_state`: 42, `n_jobs`: -1.
* **Characteristics**: Provides reliable, low-variance predictions with intrinsic feature importance extraction, achieving $R^2 = 0.9623$.

#### 5. Ridge Regression (`Ridge`)
* **Architecture & Mechanics**:
  * Regularized Ordinary Least Squares (OLS) linear model with an $L_2$ Tikhonov weight shrinkage penalty:
    $$\min_{\mathbf{w}} \|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2 + \alpha \|\mathbf{w}\|_2^2$$
* **Hyperparameters**:
  * `alpha`: 1.5.
  * `random_state`: 42.
* **Characteristics**: Trained in just 6 milliseconds. However, its linear assumption cannot capture non-linear multiplicative chip-shearing physics and power-law exponents ($h^{1-m_c}$), leading to an $R^2$ of 0.8391 and a higher error of $1.634\text{ kW}$.

---

## 3. Machining Physics & Mathematical Modeling

Unlike synthetic datasets generated with pure random noise, this system is grounded in classic manufacturing mechanics.

### 3.1 Kienzle Tangential Cutting Force Equation
The primary cutting force acting on the tool edge is modeled as:
$$F_c = k_{c1.1} \cdot a_p \cdot h^{(1 - m_c)} \cdot K_{\text{wear}} \cdot K_{\text{coolant}} \cdot K_{\gamma} \quad [\text{N}]$$

Where:
* $k_{c1.1}$: Specific cutting force at unit uncut chip thickness ($h = 1\text{ mm}$).
* $m_c$: Kienzle exponent representing the chip thickness softening/hardening behavior.
* $a_p$: Axial depth of cut in millimeters.
* $h$: Mean uncut chip thickness:
  * For CNC Milling: $h_m = f_z \cdot \sqrt{\frac{a_e}{D}}$
  * For CNC Turning: $h = f \cdot \sin(\kappa)$ ($\kappa = 75^\circ$)
* $K_{\text{wear}}$: Tool flank wear degradation multiplier:
  $$K_{\text{wear}} = 1.0 + 1.25 \cdot \left(\frac{VB}{0.30}\right)^{1.15}$$
* $K_{\text{coolant}}$: Coolant friction mitigation coefficient:
  * Dry: $1.12$
  * Flood Coolant: $0.92$
  * MQL (Minimum Quantity Lubrication): $0.96$
  * Cryogenic ($LN_2$): $0.88$
* $K_{\gamma}$: Tool rake angle correction factor:
  $$K_{\gamma} = 1.0 - 0.014 \cdot \gamma$$

### 3.2 Material Physical Constants

| Workpiece Material | $k_{c1.1}$ ($\text{N/mm}^2$) | Kienzle Exponent $m_c$ | Hardness Range (HB) | Wear Rate Factor | Speed Factor |
|---|:---:|:---:|:---:|:---:|:---:|
| **AISI 1045 Carbon Steel** | 1800.0 | 0.25 | 180 – 240 | 1.00 | 1.00 |
| **AISI 304 Stainless Steel** | 2150.0 | 0.24 | 175 – 230 | 1.25 | 0.85 |
| **Ti-6Al-4V Titanium** | 2750.0 | 0.22 | 310 – 365 | 1.60 | 0.55 |
| **Al 6061-T6 Aluminum** | 780.0 | 0.28 | 85 – 115 | 0.50 | 1.80 |
| **Inconel 718 Superalloy** | 3400.0 | 0.21 | 355 – 445 | 2.10 | 0.40 |

### 3.3 Active Cutting Power Conversion
The mechanical cutting power $P_{\text{cut}}$ delivered to the cutting zone:
$$P_{\text{cut}} = \frac{F_c \cdot v_c}{60{,}000 \cdot \eta_{\text{motor}}} \quad [\text{kW}]$$
Where $\eta_{\text{motor}} \approx 0.86$ represents the electro-mechanical efficiency of the spindle drive.

### 3.4 Multi-Subsystem Total Power Model
Total active machine electrical demand is the superposition of cutting and auxiliary active subsystems:
$$P_{\text{total}} = P_{\text{base}} + P_{\text{spindle}}(N) + P_{\text{feed}}(v_f) + P_{\text{coolant}} + P_{\text{cut}} + \epsilon_{\text{sensor}}$$

* **CNC Base Controller & Hydraulics**: $P_{\text{base}} = 0.50\text{ kW}$
* **Spindle No-Load Rotation Power**:
  $$P_{\text{spindle}} = 0.45 + 0.00030 \cdot N + 3.8 \times 10^{-8} \cdot N^2 \quad [\text{kW}]$$
* **Feed Axis Kinematic Power**:
  $$P_{\text{feed}} = 0.15 + 0.00016 \cdot v_f \quad [\text{kW}]$$
* **Coolant Pump Power**:
  * Dry: $0.00\text{ kW}$
  * MQL: $0.35\text{ kW}$
  * Flood Coolant: $1.35\text{ kW}$
  * Cryogenic Delivery: $0.85\text{ kW}$
* **Sensor Noise**: $\epsilon_{\text{sensor}} \sim \mathcal{N}(0, 0.08^2)$

### 3.5 Specific Energy Consumption ($SEC$)
$$SEC = \frac{P_{\text{total}} \times 1000}{MRR_{\text{mm}^3/\text{s}}} \quad \left[\frac{\text{J}}{\text{mm}^3}\right]$$
Where volumetric Material Removal Rate is:
* Milling: $MRR = \frac{a_p \cdot a_e \cdot v_f}{1000} \quad [\text{cm}^3/\text{min}]$
* Turning: $MRR = v_c \cdot a_p \cdot f \quad [\text{cm}^3/\text{min}]$

---

## 4. Dataset Architecture & Specifications

The dataset comprises **12,500 observation records** spanning **24 physical parameters**.

### 4.1 Feature Schema

| Column Name | Category | Range / Categories | Description |
|---|---|---|---|
| `Operation_Type` | Categorical | `CNC Milling`, `CNC Turning` | Kinematic process type |
| `Workpiece_Material` | Categorical | 5 Alloys (Steel, Stainless, Ti, Al, Inconel) | Material specification |
| `Material_Hardness_HB` | Numerical | 85 – 450 HB | Brinell hardness number |
| `Tool_Diameter_mm` | Numerical | 6.0 – 110.0 mm | Cutter or workpiece diameter |
| `Number_of_Flutes` | Discrete | 1, 2, 3, 4, 5, 6 | Tooth count ($z$) |
| `Tool_Coating` | Categorical | `TiAlN`, `TiCN`, `AlCrN`, `Uncoated Carbide` | Protective PVD/CVD coating |
| `Tool_Wear_VB_mm` | Numerical | 0.000 – 0.450 mm | Flank wear land width $VB$ |
| `Rake_Angle_deg` | Numerical | -6.0° to +12.0° | Tool rake angle $\gamma$ |
| `Cutting_Speed_vc_mpm` | Numerical | 25.0 – 480.0 m/min | Cutting surface speed $v_c$ |
| `Spindle_Speed_RPM` | Numerical | 250 – 18,000 RPM | Spindle rotational speed $N$ |
| `Feed_Rate_mm_rev` | Numerical | 0.03 – 0.40 mm/rev | Feed per revolution or tooth |
| `Axial_Depth_ap_mm` | Numerical | 0.5 – 5.0 mm | Axial depth of cut $a_p$ |
| `Radial_Depth_ae_mm` | Numerical | 0.5 – 32.0 mm | Radial cutting width $a_e$ |
| `Feed_Speed_vf_mmpm` | Numerical | 50 – 4,500 mm/min | Table/axis travel speed $v_f$ |
| `Coolant_Condition` | Categorical | `Flood Coolant`, `MQL`, `Cryogenic`, `Dry` | Coolant lubrication mode |
| `Coolant_Flow_Rate_Lpm` | Numerical | 0.0 – 35.0 L/min | Fluid volume delivery rate |
| `Material_Removal_Rate_cm3_min` | Numerical | 1.0 – 250.0 cm³/min | Volumetric chip removal rate |
| **`Power_Consumption_kW`** | **Target** | **1.44 – 29.97 kW** | **Active electric power $P_{\text{total}}$** |
| `Specific_Energy_J_mm3` | Derived | 0.8 – 45.0 J/mm³ | Physical energy intensity |

---

## 5. Preprocessing & Feature Engineering Pipeline

To maximize model performance and prevent data leakage, a strict Scikit-Learn `ColumnTransformer` pipeline is employed.

### 5.1 Physically Engineered Interaction Features

1. **Kinematic Power Coupling**:
   $$\text{Speed\_x\_Feed} = v_c \times f$$
   Captures the simultaneous intensification of shearing speed and uncut chip thickness.
2. **Chip Geometry Aspect Ratio**:
   $$\text{Depth\_Ratio} = \frac{a_p}{a_e + 10^{-4}}$$
   Distinguishes slotting cuts ($a_e \approx D$) from peripheral shoulder milling ($a_e \ll D$).
3. **Tooth Dynamic Loading**:
   $$\text{MRR\_per\_Flute} = \frac{MRR}{\max(z, 1)}$$
   Measures the volumetric burden placed on individual cutting edges.
4. **Normalized Tool Wear**:
   $$\text{Wear\_to\_Diameter\_Ratio} = \frac{VB}{D + 10^{-4}}$$
   Normalizes the significance of tool wear relative to tool rigidity and diameter.

### 5.2 Transformation Pipeline Architecture
* **Numerical Features (17 total)**: Scaled with `StandardScaler` ($\mu = 0, \sigma = 1$).
* **Categorical Features (4 total)**: Encoded with `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` producing 15 dummy indicator columns.
* **Leakage Prevention**: All transformations are fit **strictly** on the 80% training split (`X_train`) and applied identically to the 20% test split (`X_test`).
* **Artifacts Exported**:
  * `models/preprocessing_pipeline.pkl`: Fitted pipeline transformer.
  * `models/feature_metadata.json`: Feature mapping schema and split counts.

---

## 6. Project Architecture & Directory Layout

```
e:/Ml_program/
│
├── data/
│   ├── raw/
│   │   └── machining_power_consumption_12k.csv   # Complete 12,500 physics-based observations
│   └── processed/
│       ├── train_features.csv                   # 10,000 processed training records
│       └── test_features.csv                    # 2,500 processed testing records
│
├── models/
│   ├── best_model.pkl                            # Serialized champion Neural Net (MLP)
│   ├── preprocessing_pipeline.pkl               # Fitted StandardScaler + OneHotEncoder
│   ├── feature_metadata.json                    # Feature mapping and column metadata
│   ├── model_leaderboard.json                   # Cross-algorithm benchmark results
│   └── test_predictions.json                    # Test set predictions for diagnostics
│
├── reports/
│   ├── figures/
│   │   ├── model_comparison_bar.png             # Algorithm R² and RMSE comparison chart
│   │   ├── actual_vs_predicted.png              # 45-degree parity plot with ±10% error bands
│   │   ├── residual_distribution.png            # Residual distribution histogram & KDE
│   │   └── feature_importance.png               # Top 12 drivers of machining power
│   └── evaluation_summary.md                    # Auto-generated markdown evaluation report
│
├── src/
│   ├── __init__.py                              # Module initialization
│   ├── dataset_generator.py                     # Generates 12.5k physics-grounded records
│   ├── data_preprocessing.py                    # Feature engineering & ColumnTransformer pipeline
│   ├── model_training.py                        # Multi-algorithm benchmark & model serialization
│   ├── model_evaluation.py                      # Diagnostic plot generation & markdown reporter
│   └── predict.py                               # Vectorized CLI & API inference engine
│
├── app/
│   └── streamlit_app.py                         # Full-featured Streamlit Web Dashboard
│
├── requirements.txt                             # Pinned Python package dependencies
├── run_pipeline.py                              # Master 1-click pipeline execution script
├── run_app.bat                                  # Batch script for launching Streamlit UI
├── app.py                                       # Root entry wrapper for Streamlit
└── README.md                                    # Root repository documentation
```

---

## 7. Interactive Streamlit Web Dashboard Features

The web interface (`app/streamlit_app.py` or `app.py`) provides 5 comprehensive views:

1. **⚡ Real-Time Power Prediction**:
   * Interactive sliders for $v_c, f_z, a_p, a_e, VB, \gamma$.
   * Dropdown selectors for operation type, material alloy, tool coating, and coolant condition.
   * Real-time active power card ($P_{\text{total}}$ in kW).
   * Energy metrics: Spindle RPM, feed speed ($v_f$), $MRR$, Specific Energy Consumption ($SEC$ in $\text{J/mm}^3$), and hourly electricity cost ($/hr).
   * Dynamic regime status: Light Load, Moderate Load, or Heavy Roughing overload alert.
   * Component breakdown pie chart: Shearing action vs. Spindle friction vs. Coolant pump vs. Feed axis motors.

2. **📥 Dataset Explorer & CSV Download**:
   * One-click **"📥 Download Full Dataset as CSV (12,500+ Rows)"** button.
   * Multi-select interactive filters by operation, material, and coolant.
   * Paginated data view and descriptive statistical summary tables.

3. **📊 Model Benchmarking & Diagnostics**:
   * Interactive leaderboard table with champion highlighting.
   * Embedded high-resolution diagnostic plots (Algorithm comparison, Parity plot, Residual error distribution, Feature importance).

4. **🔬 Parameter Sensitivity Simulation (What-If Curves)**:
   * Dynamic simulation sweeping cutting speed, feed rate, axial depth, or tool wear across selected workpiece alloys.
   * Real-time Matplotlib curve rendering.

5. **📖 Machining Physics & Formulas**:
   * Embedded LaTeX mathematical documentation for Kienzle forces, specific cutting energy, and kinematic derivations.

---

## 8. CLI & API Inference Usage

Predictions can be performed from Python code or directly from the terminal.

### 8.1 Terminal CLI Command
```powershell
python src/predict.py --vc 180 --feed 0.15 --ap 3.0 --ae 10.0 --material "AISI 1045 Steel" --coolant "Flood Coolant"
```

**Terminal Output**:
```
--- Input Machining Process Parameters ---
  Operation_Type              : CNC Milling
  Workpiece_Material          : AISI 1045 Steel
  Cutting_Speed_vc_mpm        : 180.0
  Feed_Rate_mm_rev            : 0.15
  Axial_Depth_ap_mm           : 3.0
  Radial_Depth_ae_mm          : 10.0
  Tool_Diameter_mm            : 16.0
  Number_of_Flutes            : 4
  Tool_Wear_VB_mm             : 0.1
  Coolant_Condition           : Flood Coolant
  Tool_Coating                : TiAlN

==========================================
  PREDICTED ACTIVE POWER:  7.627 kW
==========================================
  Machine Load Status:     Moderate Load (Standard Roughing/Semi-Finish)
```

### 8.2 Python API Integration
```python
from src.predict import predict_power

scenario = {
    "Operation_Type": "CNC Milling",
    "Workpiece_Material": "Ti-6Al-4V Titanium",
    "Cutting_Speed_vc_mpm": 90.0,
    "Feed_Rate_mm_rev": 0.12,
    "Axial_Depth_ap_mm": 2.5,
    "Radial_Depth_ae_mm": 12.0,
    "Tool_Wear_VB_mm": 0.20,
    "Coolant_Condition": "Cryogenic (LN2)"
}

predicted_kw = predict_power(scenario)[0]
print(f"Predicted Active Power: {predicted_kw:.3f} kW")
```

---

## 9. Quickstart & Execution Guide

### Step 1: Install Dependencies
```powershell
cd e:\Ml_program
pip install -r requirements.txt
```

### Step 2: Run the End-to-End Pipeline
Generates the 12,500 record dataset, fits the preprocessor, benchmarks all 5 models, exports diagnostic figures, and runs an inference smoke test:
```powershell
python run_pipeline.py
```

### Step 3: Launch the Interactive Web Dashboard
Run any of the following:
```powershell
python -m streamlit run app.py
```
or
```powershell
.\run_app.bat
```
or
```powershell
streamlit run app/streamlit_app.py
```
The dashboard opens automatically in your web browser at `http://localhost:8501`.
