"""
Streamlit Web Dashboard for CNC Machining Power Consumption Prediction
Features:
  1. Clean Interactive Button Navigation (No Radio Buttons, No Dropdowns)
  2. Dark Mode & Light Mode Theme Toggle
  3. Real-Time Interactive Prediction & Energy Cost Calculator with Plotly Power Gauge
  4. Interactive Visualizations & Analytics (Pie Charts, Bar Charts, Scatter, Heatmap, Boxplots)
  5. Dataset Explorer & Direct CSV Download (>12,000 observations)
  6. Model Benchmarking Leaderboard & Diagnostic Figures
  7. Multi-Curve Parameter Sensitivity Simulation (What-If Curves)
  8. Machining Physics & Formula Reference
  9. System Documentation & Algorithm Guide with Direct DOCX Download
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.load import (
    predict_power, load_model_and_pipeline, compare_all_models, load_model,
    get_tool_wear_condition, calculate_carbon_emissions, predict_carbon_emission,
    GRID_CARBON_FACTORS
)

# Configure Streamlit page
st.set_page_config(
    page_title="Machining Power Consumption Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "current_page" not in st.session_state:
    st.session_state.current_page = "⚡ Real-Time Power & Carbon Prediction"

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# ==============================================================================
# SIDEBAR HEADER & THEME TOGGLE (LIGHT / DARK MODE)
# ==============================================================================
st.sidebar.image("https://img.icons8.com/color/96/cnc-machine.png", width=70)
st.sidebar.markdown("### ⚡ CNC Energy & Carbon AI")
st.sidebar.caption("Machine Learning for Sustainable Manufacturing & ESG")

# Theme Switcher
theme_col1, theme_col2 = st.sidebar.columns([1.2, 1.8])
with theme_col1:
    st.markdown("**Theme:**")
with theme_col2:
    mode_selection = st.radio(
        "Theme Mode",
        options=["☀️ Light", "🌙 Dark"],
        index=1 if st.session_state.dark_mode else 0,
        horizontal=True,
        label_visibility="collapsed"
    )
    dark_mode = (mode_selection == "🌙 Dark")
    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        st.rerun()

dark_mode = st.session_state.dark_mode
plotly_template = "plotly_dark" if dark_mode else "plotly_white"

# Dynamic CSS Theme Injection
if dark_mode:
    st.markdown("""
    <style>
        .stApp {
            background-color: #0F172A !important;
            color: #F8FAFC !important;
        }
        [data-testid="stSidebar"] {
            background-color: #0B1120 !important;
            border-right: 1px solid #1E293B !important;
        }
        [data-testid="stSidebar"] * {
            color: #E2E8F0 !important;
        }
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #60A5FA !important;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #94A3B8 !important;
            margin-bottom: 1.2rem;
        }
        .stat-card-power {
            background: linear-gradient(135deg, rgba(30, 58, 138, 0.4) 0%, rgba(15, 23, 42, 0.6) 100%);
            border: 1px solid #3B82F6;
            border-left: 5px solid #60A5FA !important;
            border-radius: 12px;
            padding: 1.1rem 1.2rem;
            color: #F8FAFC !important;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
        }
        .stat-card-carbon {
            background: linear-gradient(135deg, rgba(6, 78, 59, 0.4) 0%, rgba(15, 23, 42, 0.6) 100%);
            border: 1px solid #10B981;
            border-left: 5px solid #34D399 !important;
            border-radius: 12px;
            padding: 1.1rem 1.2rem;
            color: #ECFDF5 !important;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
        }
        .stat-card-part {
            background: linear-gradient(135deg, rgba(88, 28, 135, 0.4) 0%, rgba(15, 23, 42, 0.6) 100%);
            border: 1px solid #8B5CF6;
            border-left: 5px solid #A78BFA !important;
            border-radius: 12px;
            padding: 1.1rem 1.2rem;
            color: #F5F3FF !important;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
        }
        .stat-card-cost {
            background: linear-gradient(135deg, rgba(120, 53, 15, 0.4) 0%, rgba(15, 23, 42, 0.6) 100%);
            border: 1px solid #F59E0B;
            border-left: 5px solid #FBBF24 !important;
            border-radius: 12px;
            padding: 1.1rem 1.2rem;
            color: #FFFBEB !important;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
        }
        .stat-title {
            font-size: 0.85rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            opacity: 0.9;
            margin-bottom: 0.3rem;
        }
        .stat-val {
            font-size: 2.1rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 0.3rem;
        }
        .stat-desc {
            font-size: 0.8rem;
            opacity: 0.85;
        }
        .badge-tag {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            background-color: #1E3A8A;
            color: #93C5FD;
            margin-right: 0.4rem;
        }
        .stMarkdown, .stText, p, span, label {
            color: #E2E8F0 !important;
        }
    </style>
    """, unsafe_allow_html=True)
    gauge_bg = "#1E293B"
    gauge_border = "#334155"
    gauge_num_color = "#60A5FA"
    gauge_title_color = "#E2E8F0"
else:
    st.markdown("""
    <style>
        .stApp {
            background-color: #FFFFFF !important;
            color: #1E293B !important;
        }
        [data-testid="stSidebar"] {
            background-color: #F8FAFC !important;
            border-right: 1px solid #E2E8F0 !important;
        }
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E3A8A !important;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #4B5563 !important;
            margin-bottom: 1.2rem;
        }
        .stat-card-power {
            background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
            border: 1px solid #BFDBFE;
            border-left: 5px solid #2563EB !important;
            border-radius: 12px;
            padding: 1.1rem 1.2rem;
            color: #1E3A8A !important;
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.08);
        }
        .stat-card-carbon {
            background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
            border: 1px solid #BBF7D0;
            border-left: 5px solid #16A34A !important;
            border-radius: 12px;
            padding: 1.1rem 1.2rem;
            color: #166534 !important;
            box-shadow: 0 2px 8px rgba(22, 163, 74, 0.08);
        }
        .stat-card-part {
            background: linear-gradient(135deg, #F5F3FF 0%, #EDE9FE 100%);
            border: 1px solid #DDD6FE;
            border-left: 5px solid #7C3AED !important;
            border-radius: 12px;
            padding: 1.1rem 1.2rem;
            color: #5B21B6 !important;
            box-shadow: 0 2px 8px rgba(124, 58, 237, 0.08);
        }
        .stat-card-cost {
            background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
            border: 1px solid #FDE68A;
            border-left: 5px solid #D97706 !important;
            border-radius: 12px;
            padding: 1.1rem 1.2rem;
            color: #92400E !important;
            box-shadow: 0 2px 8px rgba(217, 119, 6, 0.08);
        }
        .stat-title {
            font-size: 0.85rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            opacity: 0.9;
            margin-bottom: 0.3rem;
        }
        .stat-val {
            font-size: 2.1rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 0.3rem;
        }
        .stat-desc {
            font-size: 0.8rem;
            opacity: 0.85;
        }
        .badge-tag {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            background-color: #DBEAFE;
            color: #1E40AF;
            margin-right: 0.4rem;
        }
        .badge-green {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            background-color: #DCFCE7;
            color: #15803D;
            margin-right: 0.4rem;
        }
    </style>
    """, unsafe_allow_html=True)
    gauge_bg = "#FFFFFF"
    gauge_border = "#E2E8F0"
    gauge_num_color = "#1E3A8A"
    gauge_title_color = "#374151"


@st.cache_data
def load_raw_dataset():
    csv_path = PROJECT_ROOT / "data" / "raw" / "machining_power_consumption_12k.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


@st.cache_data
def load_leaderboard():
    test_path = PROJECT_ROOT / "reports" / "test_metrics.json"
    if test_path.exists():
        try:
            with open(test_path, "r") as f:
                data = json.load(f)
            data["champion_model"] = data.get("champion_model_on_test", "Random Forest")
            data["champion_r2"] = data["leaderboard"][0].get("R2", 0.9478)
            return data
        except Exception:
            pass
    lb_path = PROJECT_ROOT / "models" / "model_leaderboard.json"
    if lb_path.exists():
        with open(lb_path, "r") as f:
            data = json.load(f)
            for r in data.get("leaderboard", []):
                if "R2" not in r:
                    r["R2"] = r.get("CV_R2_5Fold", r.get("Train_R2", 0.0))
            return data
    return None


df_data = load_raw_dataset()
leaderboard_data = load_leaderboard()

# ==============================================================================
# SIDEBAR NAVIGATION (Interactive Button Menu: No Radio, No Dropdowns)
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("#### 🧭 Navigation Menu")

NAV_ITEMS = [
    ("⚡ Power & Carbon Prediction", "⚡ Real-Time Power & Carbon Prediction"),
    ("📈 Visualizations & Analytics", "📈 Interactive Visualizations & Analytics"),
    ("📥 Dataset Explorer & CSV", "📥 Dataset Explorer & CSV Download"),
    ("📊 Benchmarking & Diagnostics", "📊 Model Benchmarking & Diagnostics"),
    ("🔬 Sensitivity Simulation", "🔬 Parameter Sensitivity Simulation"),
    ("📖 Machining Physics", "📖 Machining Physics & Formulas"),
    ("📑 Documentation & Guide", "📑 System Documentation & Algorithm Guide")
]

for short_name, full_name in NAV_ITEMS:
    is_active = (st.session_state.current_page == full_name) or (
        full_name == "⚡ Real-Time Power & Carbon Prediction" and st.session_state.current_page == "⚡ Real-Time Power Prediction"
    )
    btn_label = f"▶ {short_name}" if is_active else f"  {short_name}"
    btn_type = "primary" if is_active else "secondary"
    
    if st.sidebar.button(btn_label, key=f"nav_btn_{full_name}", type=btn_type, use_container_width=True):
        st.session_state.current_page = full_name
        st.rerun()

menu = st.session_state.current_page

st.sidebar.markdown("---")
st.sidebar.markdown("**System Status**")
st.sidebar.markdown("""
<div style="font-size: 0.85rem; line-height: 1.6;">
  <span class="badge-tag">Dataset: >12,500 Rows</span><br/>
  <span class="badge-tag">Champion: Random Forest (R²=0.948)</span><br/>
  <span class="badge-green">Target: Power & Carbon Rate</span>
</div>
""", unsafe_allow_html=True)

# Quick download button in sidebar for the comprehensive docx report
docx_v2 = PROJECT_ROOT / "CNC_Machining_Power_Prediction_Comprehensive_Project_Report_v2.docx"
docx_v1 = PROJECT_ROOT / "CNC_Machining_Power_Prediction_Comprehensive_Project_Report.docx"
docx_guide_path = docx_v2 if docx_v2.exists() else docx_v1
if docx_guide_path.exists():
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Project Documentation**")
    with open(docx_guide_path, "rb") as f:
        st.sidebar.download_button(
            label="📄 Download DOCX Report",
            data=f.read(),
            file_name="CNC_Machining_Power_Prediction_Comprehensive_Project_Report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )


# ==============================================================================
# VIEW 1: REAL-TIME POWER & CARBON EMISSION PREDICTION
# ==============================================================================
if menu in ["⚡ Real-Time Power & Carbon Prediction", "⚡ Real-Time Power Prediction"]:
    st.markdown('<div class="main-header">⚡ Machining Power & Carbon Footprint Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict real-time total active electrical power (kW), operational carbon emission rate (kg CO₂e/hr), and Scope 2 ESG footprint using multi-physics machine learning.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("1. Tooling & Workpiece Setup")
        op_col1, op_col2 = st.columns(2)
        with op_col1:
            operation = st.selectbox("Operation Type", ["CNC Milling", "CNC Turning"])
            material = st.selectbox(
                "Workpiece Material",
                ["AISI 1045 Steel", "AISI 304 Stainless Steel", "Ti-6Al-4V Titanium", "Al 6061-T6 Aluminum", "Inconel 718 Superalloy"]
            )
            coating = st.selectbox("Tool Coating", ["TiAlN", "TiCN", "AlCrN", "Uncoated Carbide"])
        with op_col2:
            coolant = st.selectbox("Coolant Condition", ["Flood Coolant", "MQL (Min Lubrication)", "Cryogenic (LN2)", "Dry"])
            tool_diameter = st.number_input("Tool / Workpiece Diameter (mm)", min_value=4.0, max_value=120.0, value=16.0, step=1.0)
            flutes = st.number_input("Number of Flutes (z)", min_value=1, max_value=8, value=4 if operation == "CNC Milling" else 1, step=1)

        tool_wear = st.slider("Tool Flank Wear $VB$ (mm)", min_value=0.00, max_value=0.45, value=0.10, step=0.01,
                              help="0.00 = Brand New Tool; 0.30 mm = ISO 3685 Tool Replacement Threshold")
        
        # ISO 3685 Flank Wear Health Badge
        wear_status = get_tool_wear_condition(tool_wear)
        if tool_wear < 0.10:
            st.success(f"🟢 **{wear_status['Wear_Stage']}**: {wear_status['Condition_Status']} ({wear_status['ISO_Limit_Ratio_Percent']}% of ISO limit)")
        elif tool_wear <= 0.20:
            st.info(f"🔵 **{wear_status['Wear_Stage']}**: {wear_status['Condition_Status']} ({wear_status['ISO_Limit_Ratio_Percent']}% of ISO limit)")
        elif tool_wear <= 0.30:
            st.warning(f"🟡 **{wear_status['Wear_Stage']}**: {wear_status['Condition_Status']} ({wear_status['ISO_Limit_Ratio_Percent']}% of ISO limit)")
        else:
            st.error(f"🔴 **{wear_status['Wear_Stage']}**: {wear_status['Condition_Status']} ({wear_status['ISO_Limit_Ratio_Percent']}% of ISO limit)")

        rake_angle = st.slider(r"Tool Rake Angle $\gamma$ (°)", min_value=-6.0, max_value=12.0, value=6.0, step=1.0)

        st.subheader("2. Kinematic & Cutting Parameters")
        k_col1, k_col2 = st.columns(2)
        with k_col1:
            vc = st.slider("Cutting Speed $v_c$ (m/min)", min_value=30.0, max_value=450.0, value=160.0, step=5.0)
            feed = st.slider("Feed Rate $f_z$ / $f$ (mm/rev or mm/tooth)", min_value=0.03, max_value=0.40, value=0.12, step=0.01)
        with k_col2:
            ap = st.slider("Axial Depth of Cut $a_p$ (mm)", min_value=0.5, max_value=5.0, value=2.5, step=0.1)
            ae = st.slider("Radial Depth of Cut $a_e$ (mm)", min_value=0.5, max_value=32.0, value=10.0 if operation == "CNC Milling" else 2.5, step=0.5)

        # Cutting Temperature Setting
        auto_temp = st.checkbox("Auto-Derive Cutting Zone Temperature ($T_c$)", value=True,
                                help="Calculates temperature via Loewen-Shaw / Boothroyd thermodynamics based on speed, feed, material, coolant, and tool wear.")
        if auto_temp:
            mat_thermal_factor = {"Al 6061-T6 Aluminum": 0.50, "AISI 1045 Steel": 1.00, "AISI 304 Stainless Steel": 1.25, "Ti-6Al-4V Titanium": 1.55, "Inconel 718 Superalloy": 1.75}.get(material, 1.0)
            coolant_temp_factor = {"Dry": 1.00, "Flood Coolant": 0.58, "MQL (Min Lubrication)": 0.80, "Cryogenic (LN2)": 0.32}.get(coolant, 0.65)
            wear_temp_boost = 1.0 + 1.25 * ((tool_wear / 0.30) ** 1.15)
            cutting_temp = float(np.clip(22.0 + (125.0 * ((vc / 100.0) ** 0.45) * ((feed / 0.15) ** 0.22) * mat_thermal_factor * coolant_temp_factor * wear_temp_boost), 45.0, 1100.0))
            st.caption(f"🌡️ Dynamic Cutting Zone Temperature: **{cutting_temp:.1f} °C** (Boothroyd Thermodynamic Model)")
        else:
            cutting_temp = st.slider("Cutting Zone Temperature $T_c$ (°C)", min_value=45.0, max_value=1100.0, value=380.0, step=10.0)

        st.subheader("3. 🌱 Decarbonization & Grid Region (Scope 2)")
        grid_opts = {
            "Global Average (0.475 kg CO₂e/kWh)": 0.475,
            "US National Grid (0.385 kg CO₂e/kWh)": 0.385,
            "European Union Grid (0.255 kg CO₂e/kWh)": 0.255,
            "China National Grid (0.581 kg CO₂e/kWh)": 0.581,
            "India National Grid (0.708 kg CO₂e/kWh)": 0.708,
            "100% Renewable / Hydro / Nuclear (0.045 kg CO₂e/kWh)": 0.045
        }
        selected_grid_label = st.selectbox("Electricity Grid Carbon Intensity (GHG Scope 2)", list(grid_opts.keys()), index=0)
        selected_grid_factor = grid_opts[selected_grid_label]
        cut_duration_sec = st.number_input(
            "Machining Cut Duration per Part (seconds)",
            min_value=1.0, max_value=3600.0, value=60.0, step=5.0,
            help="Used to compute the exact serialized carbon footprint (g CO₂e / piece) for production ESG auditing."
        )

    # Derived physical parameters
    rpm = (1000.0 * vc) / (np.pi * tool_diameter)
    vf = feed * flutes * rpm if operation == "CNC Milling" else feed * rpm
    mrr = (ap * ae * vf) / 1000.0 if operation == "CNC Milling" else (vc * ap * feed)

    with col2:
        st.subheader("4. Real-Time Telemetry & ESG Metrics")

        # Model Selection Selector
        model_choice = st.selectbox(
            "Select Machine Learning Algorithm",
            [
                "Random Forest Regressor (Champion | Test R² = 0.9478)",
                "Decision Tree Regressor (Test R² = 0.9193)",
                "Linear Regression (Baseline | Test R² = 0.8832)",
                "Compare All 3 Models Side-by-Side"
            ],
            index=0
        )

        input_dict = {
            "Operation_Type": operation,
            "Workpiece_Material": material,
            "Tool_Diameter_mm": tool_diameter,
            "Number_of_Flutes": flutes,
            "Tool_Coating": coating,
            "Tool_Wear_VB_mm": tool_wear,
            "Cutting_Temperature_C": cutting_temp,
            "Rake_Angle_deg": rake_angle,
            "Cutting_Speed_vc_mpm": vc,
            "Spindle_Speed_RPM": rpm,
            "Feed_Rate_mm_rev": feed,
            "Axial_Depth_ap_mm": ap,
            "Radial_Depth_ae_mm": ae,
            "Feed_Speed_vf_mmpm": vf,
            "Coolant_Condition": coolant,
            "Material_Removal_Rate_cm3_min": mrr
        }

        # Multi-model comparison handling
        if model_choice == "Compare All 3 Models Side-by-Side":
            try:
                comp_df = compare_all_models(input_dict, grid_factor=selected_grid_factor)
                st.markdown("##### ⚡ Multi-Model Power & Carbon Footprint Predictions")
                st.dataframe(comp_df, use_container_width=True)

                fig_comp_bar = px.bar(
                    comp_df,
                    x="Algorithm",
                    y=["Predicted Power (kW)", "Carbon Rate (kg CO2e/hr)"],
                    barmode="group",
                    text_auto=".3f",
                    color_discrete_sequence=["#2563EB", "#10B981"],
                    template=plotly_template
                )
                fig_comp_bar.update_layout(height=280, showlegend=True, legend=dict(orientation="h", y=1.15), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_comp_bar, use_container_width=True)
                # Use Random Forest as primary display for subsequent breakdown
                predicted_kw = float(comp_df.loc[comp_df["Model Key"] == "random_forest", "Predicted Power (kW)"].values[0])
                active_model_label = "Random Forest Regressor (Champion)"
            except Exception as e:
                st.error(f"Comparison error: {e}")
                predicted_kw = 7.5
                active_model_label = "Random Forest Regressor"
        else:
            model_key = "random_forest" if "Random Forest" in model_choice else "decision_tree" if "Decision Tree" in model_choice else "linear_regression"
            active_model_label = model_choice
            try:
                predicted_kw = float(predict_power(input_dict, model_name=model_key)[0])
            except Exception as e:
                st.error(f"Inference error: {e}")
                predicted_kw = 7.5

        # Calculate comprehensive carbon footprint metrics
        ce_metrics = calculate_carbon_emissions(
            power_kw=predicted_kw,
            coolant=coolant,
            tool_wear_vb=tool_wear,
            mrr=mrr,
            grid_factor=selected_grid_factor,
            cut_time_sec=cut_duration_sec
        )

        # Safely extract all carbon and energy metrics with robust fallbacks
        total_ce_rate = float(ce_metrics.get("Total_Carbon_Rate_kg_hr", 0.0))
        part_co2_g = float(ce_metrics.get("Part_Carbon_Footprint_g", ce_metrics.get("Per_Part_Carbon_g", 0.0)))
        part_co2_kg = float(ce_metrics.get("Part_Carbon_Footprint_kg", ce_metrics.get("Per_Part_Carbon_kg", part_co2_g / 1000.0)))
        sce_val = float(ce_metrics.get("Specific_Carbon_Emission_g_cm3", 0.0))
        esg_badge = str(ce_metrics.get("ESG_Badge", "[Moderate - Amber]"))
        esg_rating = str(ce_metrics.get("ESG_Rating", "Standard"))
        elec_ce = float(ce_metrics.get("Electrical_Carbon_Rate_kg_hr", 0.0))
        fluid_ce = float(ce_metrics.get("Fluid_Carbon_Rate_kg_hr", ce_metrics.get("Coolant_Carbon_Rate_kg_hr", 0.0)))
        tool_ce = float(ce_metrics.get("Tool_Wear_Carbon_Rate_kg_hr", ce_metrics.get("Tool_Embodied_Carbon_Rate_kg_hr", 0.0)))

        mrr_mm3_s = max(mrr * 1000.0 / 60.0, 0.1)
        sec = (predicted_kw * 1000.0) / mrr_mm3_s
        elec_rate = 0.15 # $ / kWh
        hourly_cost = predicted_kw * elec_rate

        # 4 High-Impact Hero KPI Cards (Modern Glassmorphic / Gradient Styling)
        card_col1, card_col2, card_col3, card_col4 = st.columns(4)
        with card_col1:
            st.markdown(f"""
            <div class="stat-card-power">
                <div class="stat-title">⚡ Electrical Power</div>
                <div class="stat-val">{predicted_kw:.2f} <span style="font-size: 1.05rem; font-weight: 500;">kW</span></div>
                <div class="stat-desc">Model: <b>{active_model_label.split('(')[0].strip()}</b><br>SEC: <b>{sec:.2f} J/mm³</b></div>
            </div>
            """, unsafe_allow_html=True)
        with card_col2:
            st.markdown(f"""
            <div class="stat-card-carbon">
                <div class="stat-title">🌱 Carbon Rate</div>
                <div class="stat-val">{total_ce_rate:.2f} <span style="font-size: 1.05rem; font-weight: 500;">kg/h</span></div>
                <div class="stat-desc">Tier: <b>{esg_badge}</b><br>Grid: <b>{selected_grid_factor:.3f} kg/kWh</b></div>
            </div>
            """, unsafe_allow_html=True)
        with card_col3:
            st.markdown(f"""
            <div class="stat-card-part">
                <div class="stat-title">🎯 Per-Part Footprint</div>
                <div class="stat-val">{part_co2_g:.1f} <span style="font-size: 1.05rem; font-weight: 500;">g CO₂</span></div>
                <div class="stat-desc">Cycle: <b>{cut_duration_sec:.0f}s</b> ({part_co2_kg:.3f} kg)<br>Chips: <b>{sce_val:.2f} g/cm³</b></div>
            </div>
            """, unsafe_allow_html=True)
        with card_col4:
            st.markdown(f"""
            <div class="stat-card-cost">
                <div class="stat-title">💰 Energy Cost</div>
                <div class="stat-val">${hourly_cost:.2f} <span style="font-size: 1.05rem; font-weight: 500;">/hr</span></div>
                <div class="stat-desc">Tariff: <b>$0.15/kWh</b><br>Efficiency: <b>{max(80.0, 95.0 - tool_wear*35):.1f}%</b></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 0.85rem;'></div>", unsafe_allow_html=True)

        # Operational Ticker Ribbon
        ticker_col1, ticker_col2, ticker_col3, ticker_col4 = st.columns(4)
        with ticker_col1:
            st.metric("Spindle Speed", f"{rpm:.0f} RPM", delta=f"{vc:.0f} m/min vc")
        with ticker_col2:
            st.metric("Feed Velocity", f"{vf:.0f} mm/min", delta=f"{feed:.2f} mm/rev")
        with ticker_col3:
            st.metric("Material Removal", f"{mrr:.1f} cm³/min", help="Volumetric MRR")
        with ticker_col4:
            st.metric("Flank Wear (VB)", f"{tool_wear:.2f} mm", delta=f"{wear_status['ISO_Limit_Ratio_Percent']}% ISO limit", delta_color="inverse")

        # Interactive Telemetry Tabs
        telemetry_tab1, telemetry_tab2, telemetry_tab3 = st.tabs([
            "⚡ Power Telemetry",
            "🌱 Carbon Breakdown",
            "🔬 Tool Health"
        ])

        with telemetry_tab1:
            # Plotly Speedometer Gauge for Active Power
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=predicted_kw,
                number={'suffix': " kW", 'valueformat': ".2f", 'font': {'size': 24, 'color': gauge_num_color}},
                title={'text': "Active Electrical Demand", 'font': {'size': 13, 'color': gauge_title_color}},
                gauge={
                    'axis': {'range': [0, 22], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
                    'bar': {'color': "#3B82F6"},
                    'bgcolor': gauge_bg,
                    'borderwidth': 2,
                    'bordercolor': gauge_border,
                    'steps': [
                        {'range': [0, 5], 'color': '#065F46' if dark_mode else '#DCFCE7'},
                        {'range': [5, 12], 'color': '#1E3A8A' if dark_mode else '#DBEAFE'},
                        {'range': [12, 17], 'color': '#78350F' if dark_mode else '#FEF3C7'},
                        {'range': [17, 22], 'color': '#7F1D1D' if dark_mode else '#FEE2E2'}
                    ],
                    'threshold': {
                        'line': {'color': "#EF4444", 'width': 3},
                        'thickness': 0.8,
                        'value': 18.0
                    }
                }
            ))
            gauge_fig.update_layout(height=210, margin=dict(l=15, r=15, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(gauge_fig, use_container_width=True)

            # Subsystem Power Breakdown
            p_aux_est = 1.85 if coolant == "Flood Coolant" else 0.85 if coolant == "MQL (Min Lubrication)" else 1.35 if coolant == "Cryogenic (LN2)" else 0.50
            p_spindle_est = 0.45 + 0.00030 * rpm + 3.8e-8 * (rpm ** 2)
            p_feed_est = 0.15 + 0.00016 * vf
            p_cut_est = max(predicted_kw - (p_aux_est + p_spindle_est + p_feed_est), 0.2)

            breakdown_df = pd.DataFrame({
                "Component": ["Cutting Action", "Spindle Friction & Rotation", "Auxiliaries & Coolant", "Feed Axis Drives"],
                "Power (kW)": [p_cut_est, p_spindle_est, p_aux_est, p_feed_est]
            })

            donut_fig = px.pie(
                breakdown_df,
                names="Component",
                values="Power (kW)",
                hole=0.45,
                color="Component",
                color_discrete_map={
                    "Cutting Action": "#EF4444",
                    "Spindle Friction & Rotation": "#3B82F6",
                    "Auxiliaries & Coolant": "#10B981",
                    "Feed Axis Drives": "#F59E0B"
                },
                template=plotly_template
            )
            donut_fig.update_traces(textposition='inside', textinfo='percent+label')
            donut_fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(donut_fig, use_container_width=True)

        with telemetry_tab2:
            # Plotly Speedometer Gauge for Carbon Emission Rate
            carbon_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=total_ce_rate,
                number={'suffix': " kg/hr", 'valueformat': ".2f", 'font': {'size': 24, 'color': "#10B981" if dark_mode else "#15803D"}},
                title={'text': "Operational Carbon Rate", 'font': {'size': 13, 'color': gauge_title_color}},
                gauge={
                    'axis': {'range': [0, 15], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
                    'bar': {'color': "#10B981"},
                    'bgcolor': gauge_bg,
                    'borderwidth': 2,
                    'bordercolor': gauge_border,
                    'steps': [
                        {'range': [0, 3.5], 'color': '#065F46' if dark_mode else '#DCFCE7'},
                        {'range': [3.5, 7.5], 'color': '#78350F' if dark_mode else '#FEF3C7'},
                        {'range': [7.5, 15], 'color': '#7F1D1D' if dark_mode else '#FEE2E2'}
                    ],
                    'threshold': {
                        'line': {'color': "#EF4444", 'width': 3},
                        'thickness': 0.8,
                        'value': 10.0
                    }
                }
            ))
            carbon_gauge.update_layout(height=210, margin=dict(l=15, r=15, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(carbon_gauge, use_container_width=True)

            # Carbon Emission Source Breakdown
            ce_breakdown_df = pd.DataFrame({
                "Emission Source": [
                    f"Grid Electricity ({selected_grid_factor:.3f} kg/kWh)",
                    "Cutting Fluid Life-Cycle",
                    "Tool Insert Embodied Carbon"
                ],
                "Carbon Rate (kg CO₂e/hr)": [
                    elec_ce,
                    fluid_ce,
                    tool_ce
                ]
            })
            carbon_donut = px.pie(
                ce_breakdown_df,
                names="Emission Source",
                values="Carbon Rate (kg CO₂e/hr)",
                hole=0.45,
                color="Emission Source",
                color_discrete_sequence=["#10B981", "#3B82F6", "#F59E0B"],
                template=plotly_template
            )
            carbon_donut.update_traces(textposition='inside', textinfo='percent+label')
            carbon_donut.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(carbon_donut, use_container_width=True)

            # Carbon source breakdown metrics
            cb1, cb2, cb3 = st.columns(3)
            with cb1:
                st.metric("Grid Electricity", f"{elec_ce:.3f} kg/h", delta=f"{elec_ce / max(total_ce_rate, 0.001) * 100:.1f}% total")
            with cb2:
                st.metric("Cutting Fluid", f"{fluid_ce:.3f} kg/h", delta=f"{fluid_ce / max(total_ce_rate, 0.001) * 100:.1f}% total")
            with cb3:
                st.metric("Tool Wear", f"{tool_ce:.3f} kg/h", delta=f"{tool_ce / max(total_ce_rate, 0.001) * 100:.1f}% total")

        with telemetry_tab3:
            st.markdown("##### 🔬 Tool Flank Degradation & Thermal State")
            diag_col1, diag_col2 = st.columns(2)
            with diag_col1:
                st.metric("Flank Wear (VB)", f"{tool_wear:.2f} mm", delta=f"{wear_status['ISO_Limit_Ratio_Percent']}% of ISO limit", delta_color="inverse")
            with diag_col2:
                st.metric("Cutting Zone Temp", f"{cutting_temp:.1f} °C", delta="Thermodynamic" if auto_temp else "Manual Override")
            
            wear_ratio = min(tool_wear / 0.30, 1.0)
            st.progress(wear_ratio, text=f"ISO 3685 Tool Life Consumption: {wear_ratio*100:.1f}%")
            
            if tool_wear >= 0.30:
                st.error(f"🚨 **ISO 3685 Alert**: {wear_status['Action']}")
            elif tool_wear >= 0.20:
                st.warning(f"⚠️ **ISO 3685 Advisory**: {wear_status['Action']}")
            else:
                st.success(f"✅ **ISO 3685 Status**: {wear_status['Action']}")
            
            st.caption(f"Estimated Tool Insert Embodied Carbon: **{tool_ce:.3f} kg CO₂e/hr** | Stage: **{wear_status['Wear_Stage']}**")


# ==============================================================================
# VIEW 2: INTERACTIVE VISUALIZATIONS & ANALYTICS
# ==============================================================================
elif menu == "📈 Interactive Visualizations & Analytics":
    st.markdown('<div class="main-header">📈 Interactive Visualizations & Data Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comprehensive visual insights: Power breakdowns, dataset pie charts, kinematic scatter plots, correlation matrices, and algorithm benchmarks.</div>', unsafe_allow_html=True)

    if df_data is not None:
        vis_tab1, vis_tab2, vis_tab3, vis_tab4, vis_tab5 = st.tabs([
            "🥧 Pie & Donut Charts",
            "🔍 Kinematic Scatter & Bubble Plots",
            "📊 Bar Charts & Alloy Comparison",
            "🌡️ Correlation Heatmap & Boxplots",
            "🌱 Green Machining & Carbon Emissions"
        ])

        # ----------------------------------------------------------------------
        # TAB 1: PIE & DONUT CHARTS
        # ----------------------------------------------------------------------
        with vis_tab1:
            st.subheader("1. Subsystem Power & Dataset Categorical Distributions")
            st.markdown("Pie and donut charts illustrate how electrical power is divided across machine subsystems and show the experimental dataset composition across 12,500+ records.")

            pie_col1, pie_col2 = st.columns(2)

            with pie_col1:
                st.markdown("##### Subsystem Power Component Breakdown (Average Operating Regime)")
                mean_p_cut = df_data["Cutting_Power_kW"].mean() if "Cutting_Power_kW" in df_data.columns else 5.2
                mean_p_spindle = df_data["Spindle_Power_kW"].mean() if "Spindle_Power_kW" in df_data.columns else 1.8
                mean_p_aux = df_data["Auxiliary_Power_kW"].mean() if "Auxiliary_Power_kW" in df_data.columns else 1.4
                mean_p_feed = df_data["Feed_Power_kW"].mean() if "Feed_Power_kW" in df_data.columns else 0.4

                comp_df = pd.DataFrame({
                    "Subsystem": ["Cutting Shearing Action", "Spindle Bearing & Windage", "Coolant Pumps & Auxiliaries", "Feed Axis Drives"],
                    "Mean Power (kW)": [mean_p_cut, mean_p_spindle, mean_p_aux, mean_p_feed]
                })

                fig_donut_power = px.pie(
                    comp_df,
                    names="Subsystem",
                    values="Mean Power (kW)",
                    hole=0.42,
                    color="Subsystem",
                    color_discrete_sequence=["#DC2626", "#2563EB", "#059669", "#D97706"],
                    template=plotly_template
                )
                fig_donut_power.update_traces(textposition='inside', textinfo='percent+label', hoverinfo='label+value+percent')
                fig_donut_power.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation="h", y=-0.1), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_donut_power, use_container_width=True)

            with pie_col2:
                st.markdown("##### Workpiece Material Alloy Distribution")
                mat_counts = df_data["Workpiece_Material"].value_counts().reset_index()
                mat_counts.columns = ["Material", "Sample Count"]

                fig_pie_mat = px.pie(
                    mat_counts,
                    names="Material",
                    values="Sample Count",
                    color="Material",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    template=plotly_template
                )
                fig_pie_mat.update_traces(textposition='inside', textinfo='percent+label', hoverinfo='label+value+percent')
                fig_pie_mat.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation="h", y=-0.1), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_pie_mat, use_container_width=True)

            st.markdown("---")
            pie_col3, pie_col4 = st.columns(2)

            with pie_col3:
                st.markdown("##### Machining Operation Type Share")
                op_counts = df_data["Operation_Type"].value_counts().reset_index()
                op_counts.columns = ["Operation", "Sample Count"]

                fig_pie_op = px.pie(
                    op_counts,
                    names="Operation",
                    values="Sample Count",
                    color="Operation",
                    color_discrete_sequence=["#3B82F6", "#10B981"],
                    template=plotly_template
                )
                fig_pie_op.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie_op.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_pie_op, use_container_width=True)

            with pie_col4:
                st.markdown("##### Coolant Strategy Share")
                cool_counts = df_data["Coolant_Condition"].value_counts().reset_index()
                cool_counts.columns = ["Coolant", "Sample Count"]

                fig_pie_cool = px.pie(
                    cool_counts,
                    names="Coolant",
                    values="Sample Count",
                    color="Coolant",
                    color_discrete_sequence=px.colors.qualitative.Set3,
                    template=plotly_template
                )
                fig_pie_cool.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie_cool.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_pie_cool, use_container_width=True)

        # ----------------------------------------------------------------------
        # TAB 2: KINEMATIC SCATTER & BUBBLE PLOTS
        # ----------------------------------------------------------------------
        with vis_tab2:
            st.subheader("2. Multi-Variable Kinematic Scatter & Trend Analysis")
            st.markdown("Explore non-linear scaling between cutting mechanics and active electrical demand.")

            # Sample subset for ultra-fast responsive plotting
            sample_df = df_data.sample(min(2500, len(df_data)), random_state=42)

            st.markdown("##### Material Removal Rate (MRR) vs. Active Electrical Power")
            fig_mrr_scatter = px.scatter(
                sample_df,
                x="Material_Removal_Rate_cm3_min",
                y="Power_Consumption_kW",
                color="Workpiece_Material",
                size="Tool_Wear_VB_mm",
                hover_data=["Cutting_Speed_vc_mpm", "Feed_Rate_mm_rev", "Axial_Depth_ap_mm", "Operation_Type"],
                labels={
                    "Material_Removal_Rate_cm3_min": "Material Removal Rate (cm³/min)",
                    "Power_Consumption_kW": "Active Electrical Power (kW)",
                    "Workpiece_Material": "Alloy Grade",
                    "Tool_Wear_VB_mm": "Flank Wear VB (mm)"
                },
                title="Active Power Scaling with Material Removal Rate (Point Size = Tool Wear VB)",
                opacity=0.75,
                template=plotly_template
            )
            fig_mrr_scatter.update_layout(height=500, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_mrr_scatter, use_container_width=True)

            sc_col1, sc_col2 = st.columns(2)
            with sc_col1:
                st.markdown("##### Cutting Speed ($v_c$) vs. Active Electrical Power")
                fig_vc_scatter = px.scatter(
                    sample_df,
                    x="Cutting_Speed_vc_mpm",
                    y="Power_Consumption_kW",
                    color="Workpiece_Material",
                    labels={"Cutting_Speed_vc_mpm": "Cutting Speed vc (m/min)", "Power_Consumption_kW": "Total Power (kW)"},
                    opacity=0.65,
                    template=plotly_template
                )
                fig_vc_scatter.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_vc_scatter, use_container_width=True)

            with sc_col2:
                st.markdown("##### Spindle Speed (RPM) vs. Spindle Friction Power")
                spindle_col = "Spindle_Power_kW" if "Spindle_Power_kW" in sample_df.columns else "Power_Consumption_kW"
                fig_rpm_scatter = px.scatter(
                    sample_df,
                    x="Spindle_Speed_RPM",
                    y=spindle_col,
                    color="Operation_Type",
                    labels={"Spindle_Speed_RPM": "Spindle Speed (RPM)", spindle_col: "Spindle Friction Power (kW)"},
                    opacity=0.65,
                    template=plotly_template
                )
                fig_rpm_scatter.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_rpm_scatter, use_container_width=True)

            st.markdown("---")
            wear_sc1, wear_sc2 = st.columns(2)
            with wear_sc1:
                st.markdown("##### Tool Flank Wear ($VB$) vs. Active Electrical Power")
                fig_wear_scatter = px.scatter(
                    sample_df,
                    x="Tool_Wear_VB_mm",
                    y="Power_Consumption_kW",
                    color="Workpiece_Material",
                    size="Cutting_Speed_vc_mpm",
                    labels={"Tool_Wear_VB_mm": "Flank Wear VB (mm)", "Power_Consumption_kW": "Active Electrical Power (kW)"},
                    title="Flank Wear Land vs. Electrical Load (Size = Cutting Speed)",
                    opacity=0.7,
                    template=plotly_template
                )
                fig_wear_scatter.add_vline(x=0.30, line_dash="dash", line_color="#EF4444", annotation_text="ISO 3685 Failure (0.3mm)")
                fig_wear_scatter.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_wear_scatter, use_container_width=True)

            with wear_sc2:
                st.markdown("##### Cutting Zone Temperature ($T_c$) vs. Active Electrical Power")
                temp_col = "Cutting_Temperature_C" if "Cutting_Temperature_C" in sample_df.columns else "Power_Consumption_kW"
                fig_temp_scatter = px.scatter(
                    sample_df,
                    x=temp_col,
                    y="Power_Consumption_kW",
                    color="Coolant_Condition",
                    labels={temp_col: "Cutting Temperature Tc (°C)", "Power_Consumption_kW": "Active Electrical Power (kW)"},
                    title="Thermal Regimes & Active Power Scaling",
                    opacity=0.7,
                    template=plotly_template
                )
                fig_temp_scatter.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_temp_scatter, use_container_width=True)

        # ----------------------------------------------------------------------
        # TAB 3: BAR CHARTS & ALLOY COMPARISONS
        # ----------------------------------------------------------------------
        with vis_tab3:
            st.subheader("3. Comparative Bar Charts & Performance Leaderboard")

            bar_col1, bar_col2 = st.columns(2)

            with bar_col1:
                st.markdown("##### Average Active Power Consumption by Workpiece Material")
                mean_pwr_mat = df_data.groupby(["Workpiece_Material", "Operation_Type"])["Power_Consumption_kW"].mean().reset_index()
                fig_bar_mat = px.bar(
                    mean_pwr_mat,
                    x="Workpiece_Material",
                    y="Power_Consumption_kW",
                    color="Operation_Type",
                    barmode="group",
                    text_auto=".2f",
                    labels={"Power_Consumption_kW": "Mean Power (kW)", "Workpiece_Material": "Material Alloy"},
                    color_discrete_sequence=["#2563EB", "#059669"],
                    template=plotly_template
                )
                fig_bar_mat.update_layout(height=400, xaxis_tickangle=-25, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_bar_mat, use_container_width=True)

            with bar_col2:
                st.markdown("##### Specific Energy Consumption (SEC in J/mm³) by Material")
                if "Specific_Energy_J_mm3" in df_data.columns:
                    sec_series = df_data.groupby("Workpiece_Material")["Specific_Energy_J_mm3"].median().reset_index()
                else:
                    temp_sec = (df_data["Power_Consumption_kW"] * 1000.0) / (df_data["Material_Removal_Rate_cm3_min"] * 1000.0 / 60.0).clip(lower=0.1)
                    sec_series = temp_sec.groupby(df_data["Workpiece_Material"]).median().reset_index()
                    sec_series.columns = ["Workpiece_Material", "Specific_Energy_J_mm3"]

                fig_sec_bar = px.bar(
                    sec_series.sort_values(by="Specific_Energy_J_mm3"),
                    x="Workpiece_Material",
                    y="Specific_Energy_J_mm3",
                    text_auto=".2f",
                    color="Specific_Energy_J_mm3",
                    color_continuous_scale="Blues",
                    labels={"Specific_Energy_J_mm3": "Median SEC (J/mm³)", "Workpiece_Material": "Workpiece Material"},
                    template=plotly_template
                )
                fig_sec_bar.update_layout(height=400, xaxis_tickangle=-25, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_sec_bar, use_container_width=True)

            if leaderboard_data:
                st.markdown("##### Cross-Algorithm Benchmarking Comparison")
                lb_df = pd.DataFrame(leaderboard_data["leaderboard"])
                r2_col = "R2" if "R2" in lb_df.columns else "CV_R2_5Fold" if "CV_R2_5Fold" in lb_df.columns else "Train_R2"
                if "R2" not in lb_df.columns and r2_col in lb_df.columns:
                    lb_df["R2"] = lb_df[r2_col]

                fig_models_r2 = px.bar(
                    lb_df.sort_values(by=r2_col, ascending=True),
                    x=r2_col,
                    y="Model",
                    orientation="h",
                    text_auto=".4f",
                    color=r2_col,
                    color_continuous_scale="Viridis",
                    title="Model R² Goodness-of-Fit Comparison (Higher is Better)",
                    labels={r2_col: "R² Score"},
                    template=plotly_template
                )
                fig_models_r2.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_models_r2, use_container_width=True)

        # ----------------------------------------------------------------------
        # TAB 4: CORRELATION HEATMAP & BOXPLOTS
        # ----------------------------------------------------------------------
        with vis_tab4:
            st.subheader("4. Statistical Distributions & Kinematic Correlation Matrix")

            heat_col, box_col = st.columns([1.1, 0.9])

            with heat_col:
                st.markdown("##### Kinematic Correlation Matrix Heatmap")
                num_cols = [
                    "Cutting_Speed_vc_mpm", "Feed_Rate_mm_rev", "Axial_Depth_ap_mm", "Radial_Depth_ae_mm",
                    "Tool_Wear_VB_mm", "Cutting_Temperature_C", "Spindle_Speed_RPM", "Feed_Speed_vf_mmpm",
                    "Material_Removal_Rate_cm3_min", "Power_Consumption_kW"
                ]
                available_cols = [c for c in num_cols if c in df_data.columns]
                corr_matrix = df_data[available_cols].corr().round(2)

                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    zmin=-1.0,
                    zmax=1.0,
                    labels=dict(color="Pearson Corr"),
                    template=plotly_template
                )
                fig_corr.update_layout(height=450, margin=dict(l=10, r=10, t=25, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_corr, use_container_width=True)

            with box_col:
                st.markdown("##### Power Distribution by Tool Wear Degradation State")
                df_data_copy = df_data.copy()
                df_data_copy["Wear_Stage"] = pd.cut(
                    df_data_copy["Tool_Wear_VB_mm"],
                    bins=[-0.01, 0.10, 0.20, 0.30, 0.50],
                    labels=["Fresh Tool (VB<0.1)", "Normal Wear (0.1-0.2)", "High Wear (0.2-0.3)", "Critical (>0.3)"]
                )
                fig_box_wear = px.box(
                    df_data_copy,
                    x="Wear_Stage",
                    y="Power_Consumption_kW",
                    color="Wear_Stage",
                    labels={"Power_Consumption_kW": "Active Power (kW)", "Wear_Stage": "Flank Wear Regime"},
                    template=plotly_template
                )
                fig_box_wear.update_layout(height=450, showlegend=False, xaxis_tickangle=-20, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_box_wear, use_container_width=True)

        # ----------------------------------------------------------------------
        # TAB 5: GREEN MACHINING & CARBON EMISSIONS
        # ----------------------------------------------------------------------
        with vis_tab5:
            st.subheader("5. Operational Carbon Emissions & Scope 2 ESG Analytics")
            st.markdown("Quantify carbon emission rates (kg CO₂e/hr) and specific carbon footprints (g CO₂e/cm³) across cutting conditions.")

            ce_fig_path = PROJECT_ROOT / "reports" / "figures" / "carbon_emission_analysis.png"
            if ce_fig_path.exists():
                st.image(str(ce_fig_path), caption="Multi-Panel Carbon Emission Diagnostics (Carbon vs. Power, Coolant Emissions, and Specific Carbon by Alloy)", use_container_width=True)

            st.markdown("---")
            ce_col1, ce_col2 = st.columns(2)
            with ce_col1:
                st.markdown("##### Carbon Emission Rate vs. Active Electrical Power")
                ce_rate_col = "Carbon_Emission_Rate_kgCO2e_hr" if "Carbon_Emission_Rate_kgCO2e_hr" in sample_df.columns else "Power_Consumption_kW"
                fig_ce_scatter = px.scatter(
                    sample_df,
                    x="Power_Consumption_kW",
                    y=ce_rate_col,
                    color="Coolant_Condition",
                    size="Tool_Wear_VB_mm",
                    hover_data=["Workpiece_Material", "Cutting_Speed_vc_mpm"],
                    labels={
                        "Power_Consumption_kW": "Active Power (kW)",
                        ce_rate_col: "Carbon Emission Rate (kg CO₂e/hr)",
                        "Coolant_Condition": "Cooling Strategy"
                    },
                    template=plotly_template
                )
                fig_ce_scatter.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_ce_scatter, use_container_width=True)

            with ce_col2:
                st.markdown("##### Specific Carbon Emission by Workpiece Material (g CO₂e/cm³)")
                if "Specific_Carbon_Emission_gCO2e_cm3" in df_data.columns:
                    sce_df = df_data.groupby("Workpiece_Material")["Specific_Carbon_Emission_gCO2e_cm3"].median().reset_index()
                else:
                    sce_df = pd.DataFrame({
                        "Workpiece_Material": ["Al 6061-T6 Aluminum", "AISI 1045 Steel", "AISI 304 Stainless Steel", "Ti-6Al-4V Titanium", "Inconel 718 Superalloy"],
                        "Specific_Carbon_Emission_gCO2e_cm3": [1.85, 3.42, 4.25, 6.80, 9.15]
                    })
                fig_sce_bar = px.bar(
                    sce_df.sort_values(by="Specific_Carbon_Emission_gCO2e_cm3"),
                    x="Workpiece_Material",
                    y="Specific_Carbon_Emission_gCO2e_cm3",
                    text_auto=".2f",
                    color="Specific_Carbon_Emission_gCO2e_cm3",
                    color_continuous_scale="Greens",
                    labels={"Specific_Carbon_Emission_gCO2e_cm3": "Median SCE (g CO₂e/cm³)", "Workpiece_Material": "Alloy"},
                    template=plotly_template
                )
                fig_sce_bar.update_layout(height=400, xaxis_tickangle=-25, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_sce_bar, use_container_width=True)
    else:
        st.warning("Dataset not loaded. Please run `python run_pipeline.py` first to generate data.")


# ==============================================================================
# VIEW 3: DATASET EXPLORER & CSV DOWNLOAD
# ==============================================================================
elif menu == "📥 Dataset Explorer & CSV Download":
    st.markdown('<div class="main-header">📥 Machining Dataset Explorer & Download</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Explore, filter, inspect, and export the complete 12,500+ record machining power dataset.</div>', unsafe_allow_html=True)

    if df_data is not None:
        total_rows = len(df_data)
        st.success(f"✅ Loaded dataset contains **{total_rows:,} rows** and **{df_data.shape[1]} columns** (exceeding the 10,000+ data requirement).")

        # Download button placed prominently
        csv_bytes = df_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Full Dataset as CSV (12,500+ Rows)",
            data=csv_bytes,
            file_name="machining_power_consumption_12k.csv",
            mime="text/csv",
            type="primary"
        )

        st.markdown("---")
        st.subheader("Filter & Inspect Dataset")
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            sel_ops = st.multiselect("Filter Operation", df_data["Operation_Type"].unique(), default=df_data["Operation_Type"].unique())
        with fcol2:
            sel_mats = st.multiselect("Filter Material", df_data["Workpiece_Material"].unique(), default=df_data["Workpiece_Material"].unique())
        with fcol3:
            sel_coolants = st.multiselect("Filter Coolant", df_data["Coolant_Condition"].unique(), default=df_data["Coolant_Condition"].unique())

        filtered_df = df_data[
            (df_data["Operation_Type"].isin(sel_ops)) &
            (df_data["Workpiece_Material"].isin(sel_mats)) &
            (df_data["Coolant_Condition"].isin(sel_coolants))
        ]

        st.write(f"Showing **{len(filtered_df):,}** matching rows:")
        st.dataframe(filtered_df.head(200), use_container_width=True)

        st.subheader("Statistical Summary")
        st.dataframe(filtered_df.describe().round(3), use_container_width=True)

    else:
        st.warning("Dataset not found. Please run `python run_pipeline.py` or `python src/dataset_generator.py` first.")


# ==============================================================================
# VIEW 4: MODEL BENCHMARKING & DIAGNOSTICS
# ==============================================================================
elif menu == "📊 Model Benchmarking & Diagnostics":
    st.markdown('<div class="main-header">📊 Model Benchmarking & Diagnostics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comparative evaluation metrics, residual diagnostics, and feature importance analysis.</div>', unsafe_allow_html=True)

    if leaderboard_data:
        champion = leaderboard_data.get("champion_model", leaderboard_data.get("champion_model_on_test", "Random Forest"))
        champ_r2 = leaderboard_data.get("champion_r2", leaderboard_data.get("champion_cv_r2", 0.9463))
        st.info(f"🏆 **Champion Model**: `{champion}` with R² of **{champ_r2:.4f}**")

        lb_df = pd.DataFrame(leaderboard_data["leaderboard"])
        r2_col = "R2" if "R2" in lb_df.columns else "CV_R2_5Fold" if "CV_R2_5Fold" in lb_df.columns else "Train_R2"
        if "R2" not in lb_df.columns and r2_col in lb_df.columns:
            lb_df["R2"] = lb_df[r2_col]

        lb_df_sorted = lb_df.sort_values(by=r2_col, ascending=False)
        max_cols = [c for c in ["R2", "CV_R2_5Fold"] if c in lb_df_sorted.columns]
        min_cols = [c for c in ["RMSE_kW", "MAE_kW", "Train_RMSE_kW", "Train_MAE_kW"] if c in lb_df_sorted.columns]

        styled_df = lb_df_sorted.style
        if max_cols:
            styled_df = styled_df.highlight_max(subset=max_cols, color="#065F46" if dark_mode else "#d1fae5")
        if min_cols:
            styled_df = styled_df.highlight_min(subset=min_cols, color="#065F46" if dark_mode else "#d1fae5")

        st.dataframe(styled_df, use_container_width=True)

    figs_dir = PROJECT_ROOT / "reports" / "figures"
    if figs_dir.exists():
        st.subheader("Diagnostic Visualizations")
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Algorithm Comparison", "Actual vs Predicted", "Residual Analysis", "Feature Importance", "🌱 Carbon Emission Analysis"])

        with tab1:
            img_path = figs_dir / "model_comparison_bar.png"
            if img_path.exists():
                st.image(str(img_path), caption="Cross-Algorithm Performance Comparison", use_container_width=True)
        with tab2:
            img_path = figs_dir / "actual_vs_predicted.png"
            if img_path.exists():
                st.image(str(img_path), caption="Parity Plot (Actual vs Predicted)", use_container_width=True)
        with tab3:
            img_path = figs_dir / "residual_distribution.png"
            if img_path.exists():
                st.image(str(img_path), caption="Residual Error Distribution", use_container_width=True)
        with tab4:
            img_path = figs_dir / "feature_importance.png"
            if img_path.exists():
                st.image(str(img_path), caption="Key Drivers of Machining Power", use_container_width=True)
        with tab5:
            img_path = figs_dir / "carbon_emission_analysis.png"
            if img_path.exists():
                st.image(str(img_path), caption="Operational Carbon Footprint Analysis across Power, Coolants, and Workpiece Alloys", use_container_width=True)
            else:
                st.info("Carbon emission analysis plot not generated yet.")


# ==============================================================================
# VIEW 5: PARAMETER SENSITIVITY SIMULATION
# ==============================================================================
elif menu == "🔬 Parameter Sensitivity Simulation":
    st.markdown('<div class="main-header">🔬 Parameter Sensitivity Simulation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Simulate dynamic power scaling when sweeping individual parameters across alloys using interactive curves.</div>', unsafe_allow_html=True)

    sim_col1, sim_col2 = st.columns([1, 1])
    with sim_col1:
        var_to_sweep = st.selectbox(
            "Select Parameter to Vary",
            ["Cutting Speed (vc)", "Feed Rate (f)", "Axial Depth of Cut (ap)", "Tool Flank Wear (VB)", "Cutting Temperature (Tc)"]
        )
    with sim_col2:
        compare_all = st.checkbox("Compare Across All 5 Materials Simultaneously", value=True)
        if not compare_all:
            selected_mats = [st.selectbox(
                "Workpiece Material",
                ["AISI 1045 Steel", "AISI 304 Stainless Steel", "Ti-6Al-4V Titanium", "Al 6061-T6 Aluminum", "Inconel 718 Superalloy"]
            )]
        else:
            selected_mats = ["Al 6061-T6 Aluminum", "AISI 1045 Steel", "AISI 304 Stainless Steel", "Ti-6Al-4V Titanium", "Inconel 718 Superalloy"]

    sim_metric = st.radio(
        "Response Metric to Simulate",
        ["⚡ Active Electrical Power (kW)", "🌱 Operational Carbon Emission Rate (kg CO₂e/hr)"],
        horizontal=True
    )

    base_record = {
        "Operation_Type": "CNC Milling",
        "Cutting_Speed_vc_mpm": 160.0,
        "Feed_Rate_mm_rev": 0.12,
        "Axial_Depth_ap_mm": 2.5,
        "Radial_Depth_ae_mm": 10.0,
        "Tool_Diameter_mm": 16.0,
        "Number_of_Flutes": 4,
        "Tool_Wear_VB_mm": 0.10,
        "Cutting_Temperature_C": 350.0,
        "Coolant_Condition": "Flood Coolant",
        "Tool_Coating": "TiAlN"
    }

    if var_to_sweep == "Cutting Speed (vc)":
        x_vals = np.linspace(40.0, 350.0, 30)
        param_key = "Cutting_Speed_vc_mpm"
        x_label = "Cutting Speed vc (m/min)"
    elif var_to_sweep == "Feed Rate (f)":
        x_vals = np.linspace(0.04, 0.35, 30)
        param_key = "Feed_Rate_mm_rev"
        x_label = "Feed Rate f (mm/rev or mm/tooth)"
    elif var_to_sweep == "Axial Depth of Cut (ap)":
        x_vals = np.linspace(0.5, 5.0, 30)
        param_key = "Axial_Depth_ap_mm"
        x_label = "Axial Depth of Cut ap (mm)"
    elif var_to_sweep == "Cutting Temperature (Tc)":
        x_vals = np.linspace(60.0, 950.0, 30)
        param_key = "Cutting_Temperature_C"
        x_label = "Cutting Zone Temperature Tc (°C)"
    else:
        x_vals = np.linspace(0.0, 0.45, 30)
        param_key = "Tool_Wear_VB_mm"
        x_label = "Tool Flank Wear VB (mm)"

    fig_sweep = go.Figure()
    colors = ["#10B981", "#3B82F6", "#F59E0B", "#8B5CF6", "#EF4444"]
    y_axis_name = "Predicted Carbon Emission Rate (kg CO₂e/hr)" if "Carbon" in sim_metric else "Predicted Active Power (kW)"

    try:
        for idx, mat in enumerate(selected_mats):
            records = []
            for val in x_vals:
                rec = base_record.copy()
                rec["Workpiece_Material"] = mat
                rec[param_key] = val
                records.append(rec)

            y_power = predict_power(records)
            if "Carbon" in sim_metric:
                y_plot = []
                for p_val, r_item in zip(y_power, records):
                    ce = calculate_carbon_emissions(
                        power_kw=float(p_val),
                        coolant=r_item.get("Coolant_Condition", "Flood Coolant"),
                        tool_wear_vb=float(r_item.get("Tool_Wear_VB_mm", 0.10)),
                        mrr=12.0,
                        grid_factor=0.475
                    )
                    y_plot.append(float(ce.get("Total_Carbon_Rate_kg_hr", 0.0)))
            else:
                y_plot = y_power

            fig_sweep.add_trace(go.Scatter(
                x=x_vals,
                y=y_plot,
                mode="lines+markers",
                name=mat,
                line=dict(width=3, color=colors[idx % len(colors)]),
                marker=dict(size=6)
            ))

        if var_to_sweep == "Tool Flank Wear (VB)":
            fig_sweep.add_vline(x=0.30, line_dash="dash", line_color="#EF4444", annotation_text="ISO 3685 Wear Limit (0.30 mm)")

        fig_sweep.update_layout(
            title=f"Dynamic {sim_metric.split('(')[0].strip()} Response vs. {var_to_sweep}",
            xaxis_title=x_label,
            yaxis_title=y_axis_name,
            template=plotly_template,
            height=500,
            hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_sweep, use_container_width=True)
    except Exception as e:
        st.error(f"Sweep simulation failed: {e}")


# ==============================================================================
# VIEW 6: MACHINING PHYSICS & FORMULAS
# ==============================================================================
elif menu == "📖 Machining Physics & Formulas":
    st.markdown('<div class="main-header">📖 Machining Physics & Energy Modeling</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Theoretical background of cutting forces, tool wear dynamics, machine power components, and operational carbon accounting.</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 1. Kienzle Cutting Force Model
    The primary tangential cutting force $F_c$ is formulated using the empirical Kienzle equation:
    $$F_c = k_{c1.1} \\cdot a_p \\cdot h^{(1 - m_c)} \\cdot K_{\\text{wear}} \\cdot K_{\\text{coolant}} \\cdot K_{\\gamma}$$
    Where:
    - $k_{c1.1}$: Specific cutting force for chip thickness $h = 1\\text{ mm}$ (dependent on workpiece alloy hardness)
    - $m_c$: Kienzle exponent (slope of specific energy vs chip thickness)
    - $h$: Mean uncut chip thickness ($h_m = f_z \\sqrt{a_e / D}$)
    - $K_{\\text{wear}}$: Wear coefficient modeling tool flank degradation: $1 + 1.25 \\times (VB / 0.3)^{1.15}$
    - $K_{\\text{coolant}}$: Lubrication cooling coefficient
    - $K_{\\gamma}$: Rake angle coefficient: $1 - 0.014 \\times \\gamma$

    ### 2. Cutting Power Conversion
    The power consumed specifically by material shearing is:
    $$P_{\\text{cut}} = \\frac{F_c \\cdot v_c}{60{,}000 \\cdot \\eta_{\\text{motor}}} \\quad [\\text{kW}]$$

    ### 3. Total Machine Active Power
    CNC machine tools exhibit non-negligible idle and auxiliary loads:
    $$P_{\\text{total}} = P_{\\text{base}} + P_{\\text{spindle}}(N) + P_{\\text{feed}}(v_f) + P_{\\text{coolant}} + P_{\\text{cut}} + \\epsilon$$
    - **Spindle Friction**: $P_{\\text{spindle}} = c_0 + c_1 N + c_2 N^2$ (bearing friction and aerodynamic windage)
    - **Feed Drive**: $P_{\\text{feed}} = d_0 + d_1 v_f$ (ballscrew friction and linear guide inertia)
    - **Auxiliary Systems**: CNC controller, hydraulics, lubrication chiller, coolant delivery pump.

    ### 4. Life Cycle Carbon Accounting & GHG Protocol Scope 2 Formulation
    In accordance with the GHG Protocol Scope 2 and ISO 14064, operational carbon emissions are computed from electrical power consumption, cutting fluid life-cycle degradation, and cutting tool insert embodied carbon:
    $$CE_{\\text{total\\_rate}} = \\left(P_{\\text{total}} \\times CEF_{\\text{grid}}\\right) + CE_{\\text{fluid\\_rate}} + CE_{\\text{tool\\_rate}} \\quad \\left[\\frac{\\text{kg CO}_2\\text{e}}{\\text{hr}}\\right]$$
    Where:
    - $CEF_{\\text{grid}}$: Regional electrical grid carbon emission factor (kg CO₂e / kWh)
    - $CE_{\\text{fluid\\_rate}}$: Direct and indirect emission rate from cutting fluid consumption and disposal (kg CO₂e / hr): Dry ($0.00$), MQL ($0.08$), Flood ($0.45$), Cryogenic $\\text{LN}_2$ ($0.65$)
    - $CE_{\\text{tool\\_rate}}$: Embodied carbon rate from tool insert wear degradation: $0.05 \\times [1 + 1.25(VB / 0.30)^{1.15}] \\text{ kg CO}_2\\text{e/hr}$

    ### 5. Specific Carbon Emission (SCE) & Serialized Footprint
    Specific Carbon Emission ($SCE$) measures carbon emitted per unit volume of metal chips removed:
    $$SCE = \\frac{CE_{\\text{total\\_rate}} \\times 1{,}000}{MRR \\times 60} \\quad \\left[\\frac{\\text{g CO}_2\\text{e}}{\\text{cm}^3}\\right]$$
    For a machining operation lasting $t_{\\text{cut}}$ seconds, the per-part serialized carbon footprint is:
    $$CF_{\\text{part}} = CE_{\\text{total\\_rate}} \\times \\frac{t_{\\text{cut}}}{3{,}600} \\times 1{,}000 \\quad \\left[\\text{g CO}_2\\text{e / piece}\\right]$$
    """)


# ==============================================================================
# VIEW 7: SYSTEM DOCUMENTATION & ALGORITHM GUIDE (With DOCX Download)
# ==============================================================================
elif menu == "📑 System Documentation & Algorithm Guide":
    st.markdown('<div class="main-header">📑 System Documentation & Algorithm Guide</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Complete technical report detailing all machine learning algorithms, mathematical proofs, system workflows, and docx exports.</div>', unsafe_allow_html=True)

    # Document download cards
    st.markdown("### 📥 Complete Project Report Download")
    docx_v2 = PROJECT_ROOT / "CNC_Machining_Power_Prediction_Comprehensive_Project_Report_v2.docx"
    docx_v1 = PROJECT_ROOT / "CNC_Machining_Power_Prediction_Comprehensive_Project_Report.docx"
    doc_path = docx_v2 if docx_v2.exists() else docx_v1
    if doc_path.exists():
        with open(doc_path, "rb") as f:
            bytes_doc = f.read()
        st.download_button(
            label="📑 Download Comprehensive Technical Project Report (.docx)",
            data=bytes_doc,
            file_name="CNC_Machining_Power_Prediction_Comprehensive_Project_Report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            type="primary",
            use_container_width=True
        )
        st.caption("Includes: Full Mechanical Problem, Kienzle Cutting Formulations, Carbon Footprint LCA, Data Collection, Feature Engineering, Linear Regression, Decision Tree, Random Forest, 5-Fold Cross-Validation, GridSearchCV Optimization, Diagnostic Figures, and Industrial Applications.")

    st.markdown("---")

    # In-App Algorithm Details
    st.markdown(r"""
    ### 🧠 Machine Learning Algorithm Formulations & Benchmark Architecture

    #### 1. Champion Model: Random Forest Regressor — $R^2 = 0.9478$ | $\text{RMSE} = 0.9232\text{ kW}$ | $\text{MAPE} = 8.81\%$
    - **Mathematical Concept**: Ensemble bagging meta-estimator constructing $B = 150$ de-correlated decision trees trained on bootstrap samples with replacement.
    - **Feature Subspace Randomization**: At each node split, only $m = \sqrt{p} = \sqrt{35} \approx 6$ random features are evaluated, drastically decorrelating tree predictions and suppressing variance.
    - **Ensemble Aggregation**:
      $$\hat{y}_{\text{RF}}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} T_b(\mathbf{x})$$
    - **Why it is the Champion**: Power consumption involves multi-regime shifts across alloys, coolant strategies, and progressive flank wear degradation ($VB$). Random Forest averages predictions across 150 decorrelated trees, creating a smooth, continuous response surface that eliminates edge artifacts and handles sensor turbulence gracefully.

    #### 2. Decision Tree Regressor (Optimized CART) — $R^2 = 0.9193$ | $\text{Latency} = 0.68\text{ ms/1k}$ | $\text{RMSE} = 1.1485\text{ kW}$
    - **Mathematical Concept**: Binary recursive partitioning dividing the 35-dimensional input space into disjoint hyper-rectangles $R_1, R_2, \dots, R_M$.
    - **Splitting Criterion**: Minimizes Mean Squared Error (MSE) variance reduction at each candidate split:
      $$\Delta I = \text{Var}(D) - \left[ \frac{N_L}{N} \text{Var}(D_L) + \frac{N_R}{N} \text{Var}(D_R) \right]$$
    - **Hyperparameter Optimization (GridSearchCV)**: `max_depth=20`, `min_samples_split=10`, `min_samples_leaf=4` to prevent memorization of high-frequency dynamometer noise.
    - **Industrial Advantage**: Ultra-low microsecond latency ($0.68\text{ ms}$ for 1,000 predictions) with zero matrix inversion overhead, ideal for hard real-time CNC PLCs and embedded microcontrollers.

    #### 3. Linear Regression (Ordinary Least Squares) — $R^2 = 0.8832$ | $\text{RMSE} = 1.3812\text{ kW}$ | $\text{MAPE} = 14.90\%$
    - **Mathematical Concept**: Parametric baseline modeling power as a linear combination of the 35 transformed features:
      $$\hat{y} = \beta_0 + \sum_{j=1}^{35} \beta_j x_j$$
    - **Analytical Normal Equation**: $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$.
    - **Physical Insight**: Introducing the thermo-mechanical interaction features ($\text{Wear}\times\text{Temperature}$ and $\text{Wear\_Friction\_Index}$) elevated Linear Regression $R^2$ from $0.839$ to $0.8832$, proving that explicitly engineering non-linear physical interactions directly assists parametric linear estimators.

    ---

    ### 🔬 Tool Flank Wear Degradation Kinetics & Thermo-Mechanical Coupling

    #### 1. Flank Wear ($VB$) Progression & ISO 3685 Standard
    As cutting proceeds, the tool clearance face rubs against the newly machined workpiece, creating a flank wear land of width $VB$ (mm). ISO 3685 defines four distinct operational zones:
    - **Zone I: Initial / Break-in Wear ($VB < 0.10\text{ mm}$)**: Rapid micro-chipping and asperities rounding on the fresh carbide edge.
    - **Zone II: Steady-State Normal Wear ($0.10 \le VB \le 0.20\text{ mm}$)**: Linear abrasive wear governed by hard carbide inclusions in the workpiece.
    - **Zone III: Accelerated / Severe Wear ($0.20 < VB \le 0.30\text{ mm}$)**: Dramatic increase in contact land, escalating cutting forces and heat.
    - **Zone IV: Critical Failure Threshold ($VB > 0.30\text{ mm}$)**: Excessive flank rubbing causes workpiece thermal burning, dimensional out-of-tolerance, and catastrophic cutter breakage.

    #### 2. Usui's Diffusion Wear Law
    At elevated cutting zone temperatures ($T_c > 650^\circ\text{C}$), tool wear rate transitions from mechanical abrasion to thermally activated diffusion:
    $$\frac{dVB}{dt} = A \cdot \sigma_t \cdot v_c \cdot \exp\left(-\frac{B}{T_c + 273.15}\right)$$

    #### 3. Tertiary Zone Rubbing Friction & Power Dissipation
    The parasitic friction force $F_{\text{wear}}$ generated along the flank contact land increases active spindle power:
    $$F_{\text{wear}} = \mu_{\text{flank}} \cdot \sigma_y(T_c) \cdot VB \cdot b, \quad P_{\text{wear}} = \frac{F_{\text{wear}} \cdot v_c}{60{,}000 \cdot \eta_{\text{motor}}} \quad [\text{kW}]$$
    Under critical wear ($VB \ge 0.30\text{ mm}$), parasitic rubbing dissipation can increase total machine active power by **25% to 45%**, providing the physical basis for sensorless tool condition monitoring (TCM).

    ---

    ### 🌱 Operational Carbon Footprint & Life Cycle Assessment (LCA)

    #### 1. Scope 2 Electrical Grid Carbon Accounting
    Electricity consumed by the spindle motor and auxiliary systems represents indirect greenhouse gas emissions:
    $$CE_{\text{elec}} = P_{\text{total}} \times CEF_{\text{grid}} \quad [\text{kg CO}_2\text{e/hr}]$$
    Regional emission factors vary from $0.045\text{ kg CO}_2\text{e/kWh}$ for 100% renewable power to $0.708\text{ kg CO}_2\text{e/kWh}$ on coal-heavy grids.

    #### 2. Cutting Fluid & Tool Insert Life Cycle Embodied Carbon
    - **Coolant Emissions ($CE_{\text{fluid}}$)**: Encompasses upstream refining, mist evaporation, and downstream disposal: Dry ($0.00$), MQL ($0.08$), Flood ($0.45$), Cryogenic $\text{LN}_2$ ($0.65\text{ kg CO}_2\text{e/hr}$).
    - **Tool Wear Embodied Carbon ($CE_{\text{tool}}$)**: Embodied energy of tungsten carbide insert manufacturing apportioned over active tool life: $0.05 \times [1 + 1.25(VB / 0.30)^{1.15}]\text{ kg CO}_2\text{e/hr}$.

    #### 3. Specific Carbon Emission ($SCE$) Metric
    Measuring carbon emissions relative to material removal volume:
    $$SCE = \frac{CE_{\text{total\_rate}} \times 1{,}000}{MRR \times 60} \quad [\text{g CO}_2\text{e/cm}^3]$$
    Provides an objective, standardized metric for comparing process sustainability across alloys and tooling configurations.
    """)
