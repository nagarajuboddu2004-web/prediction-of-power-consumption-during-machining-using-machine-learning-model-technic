# ⚡ CNC Machining Power Consumption Prediction using Machine Learning

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost%20%7C%20Neural%20Net-green.svg)](https://xgboost.readthedocs.io/)

An end-to-end industrial Machine Learning solution for predicting total electrical power consumption ($P_{\text{total}}$ in kW) during CNC machining operations (Milling and Turning). Designed for smart manufacturing, digital twins, green machining optimization, and tool condition monitoring.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Machining Physics & Mathematical Modeling](#-machining-physics--mathematical-modeling)
- [Dataset Specifications (>12,500 Observations)](#-dataset-specifications-12500-observations)
- [Machine Learning Benchmark Results](#-machine-learning-benchmark-results)
- [Project Architecture](#-project-architecture)
- [Installation & Quickstart](#-installation--quickstart)
- [Interactive Streamlit Web Dashboard](#-interactive-streamlit-web-dashboard)
- [CLI Inference Usage](#-cli-inference-usage)
- [Diagnostic Plots & Visualizations](#-diagnostic-plots--visualizations)
- [License](#-license)

---

## 🚀 Project Overview
Electrical energy consumption accounts for a significant portion of manufacturing overhead and industrial carbon emissions. Predicting machining power enables:
1. **Sustainable & Green Machining**: Selecting cutting parameters ($v_c, f_z, a_p$) that minimize Specific Energy Consumption ($SEC$).
2. **Tool Health Monitoring**: Detecting excess power draw resulting from severe tool flank wear ($VB$).
3. **Machine Protection**: Preventing spindle overload trips and tool breakage during high-speed roughing.
4. **Accurate Cost Estimation**: Real-time energy cost calculations per machined part.

---

## 🔬 Machining Physics & Mathematical Modeling

The dataset is synthesized using rigorous classical manufacturing physics rather than naive random distributions:

### 1. Kienzle Tangential Cutting Force Equation
The primary cutting force $F_c$ is modeled by:
$$F_c = k_{c1.1} \cdot a_p \cdot h^{(1 - m_c)} \cdot K_{\text{wear}} \cdot K_{\text{coolant}} \cdot K_{\gamma} \quad [\text{N}]$$

Where:
- $k_{c1.1}$: Specific cutting force for unit chip thickness $h = 1\text{ mm}$ (e.g., $1800\text{ N/mm}^2$ for AISI 1045 Steel, $3400\text{ N/mm}^2$ for Inconel 718).
- $m_c$: Kienzle exponent representing the chip thickness coefficient.
- $h$: Mean uncut chip thickness ($h_m = f_z \sqrt{a_e / D}$ for milling; $h = f \sin \kappa$ for turning).
- $K_{\text{wear}}$: Flank wear degradation factor: $K_{\text{wear}} = 1 + 1.25 \left(\frac{VB}{0.30}\right)^{1.15}$.
- $K_{\text{coolant}}$: Lubrication friction mitigation factor (Flood: 0.92, MQL: 0.96, Cryogenic: 0.88, Dry: 1.12).
- $K_{\gamma}$: Rake angle correction: $K_{\gamma} = 1 - 0.014 \cdot \gamma$.

### 2. Cutting Power Conversion
$$P_{\text{cut}} = \frac{F_c \cdot v_c}{60{,}000 \cdot \eta_{\text{motor}}} \quad [\text{kW}]$$
Where $\eta_{\text{motor}} \approx 0.86$ is the spindle electrical-to-mechanical conversion efficiency.

### 3. Machine Subsystem Active Power Breakdown
Total electrical power draw $P_{\text{total}}$ includes idle and auxiliary components:
$$P_{\text{total}} = P_{\text{base}} + P_{\text{spindle}}(N) + P_{\text{feed}}(v_f) + P_{\text{coolant}} + P_{\text{cut}} + \epsilon_{\text{sensor}}$$

- **Spindle No-Load Rotation Power**: $P_{\text{spindle}} = 0.45 + 0.00030 \cdot N + 3.8 \times 10^{-8} \cdot N^2$ [kW]
- **Feed Axis Kinematic Power**: $P_{\text{feed}} = 0.15 + 0.00016 \cdot v_f$ [kW]
- **Coolant Delivery Power**: $P_{\text{coolant}} \in \{0.00\text{ (Dry)}, 0.35\text{ (MQL)}, 1.35\text{ (Flood)}, 0.85\text{ (LN2)}\}$ [kW]
- **Base Controller Power**: $P_{\text{base}} = 0.50\text{ kW}$ (CNC controller, hydraulics, fans)

---

## 📊 Dataset Specifications (>12,500 Observations)

The dataset contains **12,500 rows** across **24 columns**, covering 5 industrial alloys and 2 primary machining processes.

### Schema Overview:
| Column Name | Data Type | Physical Range / Values | Description |
|---|---|---|---|
| `Operation_Type` | Categorical | `CNC Milling`, `CNC Turning` | Machining kinematic process |
| `Workpiece_Material` | Categorical | 5 Alloys (Steel, Stainless, Ti, Al, Inconel) | Material grade |
| `Material_Hardness_HB` | Numerical | 85 – 450 HB | Brinell hardness number |
| `Tool_Diameter_mm` | Numerical | 6.0 – 110.0 mm | Cutter or workpiece diameter |
| `Number_of_Flutes` | Integer | 1, 2, 3, 4, 5, 6 | Cutter tooth count ($z$) |
| `Tool_Coating` | Categorical | `TiAlN`, `TiCN`, `AlCrN`, `Uncoated` | PVD/CVD protective coating |
| `Tool_Wear_VB_mm` | Numerical | 0.000 – 0.450 mm | Flank wear width $VB$ |
| `Rake_Angle_deg` | Numerical | -6.0° to +12.0° | Tool rake angle |
| `Cutting_Speed_vc_mpm`| Numerical | 25.0 – 480.0 m/min | Cutting speed $v_c$ |
| `Spindle_Speed_RPM` | Numerical | 250 – 18,000 RPM | Spindle rotational speed $N$ |
| `Feed_Rate_mm_rev` | Numerical | 0.03 – 0.40 mm/rev | Feed per tooth or revolution |
| `Axial_Depth_ap_mm` | Numerical | 0.5 – 5.0 mm | Axial depth of cut $a_p$ |
| `Radial_Depth_ae_mm` | Numerical | 0.5 – 32.0 mm | Radial depth / engagement $a_e$ |
| `Feed_Speed_vf_mmpm` | Numerical | 50 – 4,500 mm/min | Axis linear feed speed $v_f$ |
| `Coolant_Condition` | Categorical | `Flood`, `MQL`, `Cryogenic`, `Dry` | Coolant lubrication mode |
| `Coolant_Flow_Rate_Lpm`| Numerical | 0.0 – 35.0 L/min | Fluid volume flow rate |
| `Material_Removal_Rate_cm3_min` | Numerical | 1.0 – 250.0 cm³/min | Volumetric rate of cut ($MRR$) |
| `Power_Consumption_kW` | **Target** | 1.44 – 29.97 kW | **Total active power ($P_{\text{total}}$)** |
| `Specific_Energy_J_mm3`| Numerical | 0.8 – 45.0 J/mm³ | Energy required per mm³ chip |

---

## 🏆 Machine Learning Benchmark Results

Trained on 10,000 observations and evaluated on 2,500 held-out test observations:

| Model | $R^2$ Score | RMSE (kW) | MAE (kW) | MAPE (%) | Train Time |
|---|---|---|---|---|---|
| 🥇 **Neural Network (MLP)** | **0.9872** | **0.461 kW** | **0.321 kW** | **4.60%** | 4.80 s |
| 🥈 **HistGradientBoosting** | **0.9792** | **0.588 kW** | **0.395 kW** | **5.46%** | 2.16 s |
| 🥉 **XGBoost Regressor** | **0.9787** | **0.594 kW** | **0.395 kW** | **5.36%** | 0.96 s |
| 4. **Random Forest** | 0.9623 | 0.791 kW | 0.523 kW | 7.28% | 3.16 s |
| 5. **Ridge Regression** | 0.8391 | 1.634 kW | 1.144 kW | 17.55% | 0.01 s |

---

## 📁 Project Architecture

```
e:/Ml_program/
│
├── data/
│   ├── raw/
│   │   └── machining_power_consumption_12k.csv   # Complete >12,500 record dataset
│   └── processed/
│       ├── train_features.csv                   # Transformed training features
│       └── test_features.csv                    # Transformed testing features
│
├── models/
│   ├── best_model.pkl                            # Serialized champion model (MLP / XGBoost)
│   ├── preprocessing_pipeline.pkl               # OneHotEncoder + RobustScaler pipeline
│   ├── feature_metadata.json                    # Feature mapping & metadata
│   ├── model_leaderboard.json                   # Cross-algorithm benchmark results
│   └── test_predictions.json                    # Test set predictions for diagnostics
│
├── reports/
│   ├── figures/
│   │   ├── model_comparison_bar.png             # Algorithm comparison horizontal bars
│   │   ├── actual_vs_predicted.png              # 45-degree parity plot with ±10% band
│   │   ├── residual_distribution.png            # Residual error histogram & KDE
│   │   └── feature_importance.png               # Top 12 drivers of machining power
│   └── evaluation_summary.md                    # Structured markdown evaluation report
│
├── src/
│   ├── __init__.py
│   ├── dataset_generator.py                     # Generates 12.5k physics-based records
│   ├── data_preprocessing.py                    # Cleaning, feature engineering, pipeline
│   ├── model_training.py                        # Multi-model training and benchmarking
│   ├── model_evaluation.py                      # Visualization and report generation
│   └── predict.py                               # CLI inference module
│
├── app/
│   └── streamlit_app.py                         # Interactive web dashboard & CSV downloader
│
├── requirements.txt                             # Python package dependencies
├── run_pipeline.py                              # One-click master pipeline runner
└── README.md                                    # Comprehensive documentation
```

---

## 💻 Installation & Quickstart

### 1. Clone or Open the Workspace
```powershell
cd e:\Ml_program
```

### 2. Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Run the End-to-End Pipeline
Execute the full workflow (generates dataset, trains models, generates plots, and runs a smoke test):
```powershell
python run_pipeline.py
```

---

## ⚡ Interactive Streamlit Web Dashboard

Launch the interactive web UI using any of these commands:
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

### Key Features of the Dashboard:
1. **⚡ Real-Time Power Prediction**:
   - Interactive sliders for Cutting Speed ($v_c$), Feed Rate ($f$), Axial Depth ($a_p$), Radial Depth ($a_e$), and Tool Flank Wear ($VB$).
   - Live instantaneous active power (kW) estimation.
   - Specific Energy Consumption ($SEC$) in $\text{J/mm}^3$.
   - Operating regime classification (Light Load / Eco, Moderate, Heavy Roughing).
   - Estimated hourly electricity cost ($/hr).
   - Pie chart showing power breakdown (Shearing, Spindle friction, Coolant pump, Feed axes).

2. **📥 Dataset Explorer & CSV Download**:
   - Direct one-click **"📥 Download Full Dataset as CSV (12,500+ Rows)"** button.
   - Dynamic multi-select filters by material, operation, and coolant.
   - Interactive data table and descriptive statistics.

3. **📊 Model Benchmarking & Diagnostics**:
   - Live model leaderboard table highlighting champion scores.
   - Tabs displaying high-resolution diagnostic plots.

4. **🔬 Parameter Sensitivity Simulation**:
   - What-If curve generator: dynamically simulates how power responds when sweeping cutting speed, feed rate, or tool wear across alloy grades.

5. **📖 Machining Physics & Formulas**:
   - Embedded LaTeX equations for Kienzle forces, specific cutting energy, and kinematic derivations.

---

## 🖥️ CLI Inference Usage

Predict power directly from terminal commands:

```powershell
python src/predict.py --vc 180 --feed 0.15 --ap 3.0 --ae 10.0 --material "AISI 1045 Steel" --coolant "Flood Coolant"
```

**Output:**
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

---

## 📈 Diagnostic Plots & Visualizations

The pipeline automatically outputs publication-quality figures in `reports/figures/`:
- **`model_comparison_bar.png`**: Side-by-side comparison of $R^2$ scores and RMSE across all algorithms.
- **`actual_vs_predicted.png`**: Parity scatter plot with the $y=x$ ideal line and $\pm 10\%$ error bounds.
- **`residual_distribution.png`**: Residual error histogram demonstrating zero-mean normal distribution with no heteroscedastic drift.
- **`feature_importance.png`**: Relative weights of top 12 drivers (Material Removal Rate, Cutting Speed, Axial Depth, Material Hardness, Tool Wear).

---

## 📜 License
This project is open source and available under the [MIT License](LICENSE).
#   p r e d i c t i o n - o f - p o w e r - c o n s u m p t i o n - d u r i n g - m a c h i n i n g - u s i n g - m a c h i n e - l e a r n i n g - m o d e l - t e c h n i c  
 #   p r e d i c t i o n - o f - p o w e r - c o n s u m p t i o n - d u r i n g - m a c h i n i n g - u s i n g - m a c h i n e - l e a r n i n g - m o d e l - t e c h n i c  
 