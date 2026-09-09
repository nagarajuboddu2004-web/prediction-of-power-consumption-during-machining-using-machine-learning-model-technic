"""
Physics-Based Machining Power Consumption Dataset Generator
Generates realistic multi-operation (Milling & Turning) dataset based on 
Kienzle cutting force equations, tool wear degradation, machine kinematics, 
and auxiliary power consumption.
"""

import os
import argparse
import numpy as np
import pandas as pd

# Physical constants and material properties
MATERIAL_PROPERTIES = {
    "AISI 1045 Steel": {
        "kc1_1": 1800.0,      # Specific cutting force at h=1mm (N/mm^2)
        "mc": 0.25,           # Kienzle exponent
        "hardness_range": (180, 240), # Brinell Hardness HB
        "base_wear_rate": 1.0,
        "speed_factor": 1.0,
    },
    "AISI 304 Stainless Steel": {
        "kc1_1": 2150.0,
        "mc": 0.24,
        "hardness_range": (175, 230),
        "base_wear_rate": 1.25,
        "speed_factor": 0.85,
    },
    "Ti-6Al-4V Titanium": {
        "kc1_1": 2750.0,
        "mc": 0.22,
        "hardness_range": (310, 365),
        "base_wear_rate": 1.60,
        "speed_factor": 0.55,
    },
    "Al 6061-T6 Aluminum": {
        "kc1_1": 780.0,
        "mc": 0.28,
        "hardness_range": (85, 115),
        "base_wear_rate": 0.50,
        "speed_factor": 1.80,
    },
    "Inconel 718 Superalloy": {
        "kc1_1": 3400.0,
        "mc": 0.21,
        "hardness_range": (355, 445),
        "base_wear_rate": 2.10,
        "speed_factor": 0.40,
    }
}

COOLANT_SPECS = {
    "Dry": {"force_factor": 1.12, "pump_power_kw": 0.00, "flow_range": (0.0, 0.0)},
    "Flood Coolant": {"force_factor": 0.92, "pump_power_kw": 1.35, "flow_range": (15.0, 35.0)},
    "MQL (Min Lubrication)": {"force_factor": 0.96, "pump_power_kw": 0.35, "flow_range": (0.05, 0.25)},
    "Cryogenic (LN2)": {"force_factor": 0.88, "pump_power_kw": 0.85, "flow_range": (2.0, 6.0)}
}

TOOL_COATINGS = {
    "Uncoated Carbide": {"friction_factor": 1.10, "wear_multiplier": 1.30},
    "TiAlN": {"friction_factor": 0.95, "wear_multiplier": 0.85},
    "TiCN": {"friction_factor": 0.92, "wear_multiplier": 0.90},
    "AlCrN": {"friction_factor": 0.88, "wear_multiplier": 0.80}
}


def generate_machining_dataset(num_samples: int = 12500, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates a physics-grounded dataset for power consumption during machining.
    
    Parameters:
        num_samples: Number of records to generate (default: 12500)
        random_seed: Reproducibility seed
        
    Returns:
        pd.DataFrame containing machining process parameters and power consumption
    """
    np.random.seed(random_seed)
    
    # 1. Operation Type & Material
    operations = np.random.choice(["CNC Milling", "CNC Turning"], size=num_samples, p=[0.55, 0.45])
    material_names = list(MATERIAL_PROPERTIES.keys())
    materials = np.random.choice(material_names, size=num_samples, p=[0.30, 0.25, 0.15, 0.20, 0.10])
    
    coolant_types = list(COOLANT_SPECS.keys())
    coolants = np.random.choice(coolant_types, size=num_samples, p=[0.15, 0.50, 0.25, 0.10])
    
    coating_types = list(TOOL_COATINGS.keys())
    coatings = np.random.choice(coating_types, size=num_samples, p=[0.20, 0.35, 0.25, 0.20])
    
    # Pre-allocate output arrays
    hardness = np.zeros(num_samples)
    cutting_speed = np.zeros(num_samples)
    spindle_speed = np.zeros(num_samples)
    feed_rate = np.zeros(num_samples)
    axial_depth = np.zeros(num_samples)
    radial_depth = np.zeros(num_samples)
    tool_diameter = np.zeros(num_samples)
    num_flutes = np.zeros(num_samples, dtype=int)
    rake_angle = np.zeros(num_samples)
    tool_wear = np.zeros(num_samples)
    coolant_flow = np.zeros(num_samples)
    feed_speed = np.zeros(num_samples)
    mrr = np.zeros(num_samples)
    cutting_force = np.zeros(num_samples)
    p_spindle = np.zeros(num_samples)
    p_feed = np.zeros(num_samples)
    p_aux = np.zeros(num_samples)
    p_cutting = np.zeros(num_samples)
    total_power = np.zeros(num_samples)
    specific_energy = np.zeros(num_samples)
    
    for i in range(num_samples):
        op = operations[i]
        mat = materials[i]
        mat_prop = MATERIAL_PROPERTIES[mat]
        clnt = coolants[i]
        clnt_prop = COOLANT_SPECS[clnt]
        coat = coatings[i]
        coat_prop = TOOL_COATINGS[coat]
        
        # Hardness with mild gaussian variation
        h_min, h_max = mat_prop["hardness_range"]
        hardness[i] = np.clip(np.random.normal((h_min + h_max) / 2.0, (h_max - h_min) / 6.0), h_min - 5, h_max + 10)
        
        # Tool wear VB (0.00 to 0.42 mm, Weibull distribution)
        tool_wear[i] = np.clip(np.random.weibull(1.6) * 0.15 * mat_prop["base_wear_rate"] * coat_prop["wear_multiplier"], 0.0, 0.45)
        
        # Rake angle in degrees (-6 to +12)
        rake_angle[i] = np.random.uniform(-6.0, 10.0)
        
        # Coolant flow rate
        f_min, f_max = clnt_prop["flow_range"]
        coolant_flow[i] = np.random.uniform(f_min, f_max) if f_max > 0 else 0.0
        
        if op == "CNC Milling":
            # Tool diameter (e.g., 6, 8, 10, 12, 16, 20, 25, 32 mm)
            d_choices = np.array([6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 25.0, 32.0])
            tool_diameter[i] = np.random.choice(d_choices)
            
            # Number of flutes
            if tool_diameter[i] <= 8.0:
                num_flutes[i] = np.random.choice([2, 3, 4], p=[0.3, 0.4, 0.3])
            elif tool_diameter[i] <= 16.0:
                num_flutes[i] = np.random.choice([3, 4, 5], p=[0.2, 0.6, 0.2])
            else:
                num_flutes[i] = np.random.choice([4, 5, 6], p=[0.4, 0.3, 0.3])
                
            # Cutting speed (m/min) scaled by material machinability
            base_vc = np.random.uniform(90.0, 260.0) * mat_prop["speed_factor"]
            cutting_speed[i] = np.clip(base_vc, 25.0, 480.0)
            
            # Spindle Speed N = 1000 * vc / (pi * D)
            n_calc = (1000.0 * cutting_speed[i]) / (np.pi * tool_diameter[i])
            spindle_speed[i] = np.clip(n_calc, 400.0, 18000.0)
            
            # Feed per tooth fz (mm/tooth)
            fz = np.random.uniform(0.04, 0.22)
            feed_rate[i] = fz
            
            # Table feed speed vf = fz * z * N (mm/min)
            vf = fz * num_flutes[i] * spindle_speed[i]
            feed_speed[i] = vf
            
            # Depths of cut
            axial_depth[i] = np.random.uniform(0.5, min(4.5, tool_diameter[i] * 0.45))
            radial_depth[i] = np.random.uniform(0.5, tool_diameter[i] * 0.90)
            
            # Material Removal Rate MRR (cm^3/min)
            mrr_calc = (axial_depth[i] * radial_depth[i] * vf) / 1000.0
            mrr[i] = mrr_calc
            
            # Effective uncut chip thickness (mean chip thickness hm)
            hm = fz * np.sqrt(radial_depth[i] / tool_diameter[i])
            hm = max(hm, 0.01)
            
            # Kienzle Specific Cutting Force kc
            kc11 = mat_prop["kc1_1"] * (hardness[i] / ((h_min + h_max) / 2.0)) ** 0.5
            mc = mat_prop["mc"]
            k_gamma = 1.0 - 0.014 * rake_angle[i]
            k_wear = 1.0 + 1.25 * ((tool_wear[i] / 0.30) ** 1.15)
            k_clnt = clnt_prop["force_factor"]
            k_coat = coat_prop["friction_factor"]
            
            kc = kc11 * (hm ** (-mc)) * k_gamma * k_wear * k_clnt * k_coat
            
            # Total Tangential Cutting Force Fc (N)
            fc = kc * axial_depth[i] * hm * (num_flutes[i] * (radial_depth[i] / (np.pi * tool_diameter[i])))
            cutting_force[i] = np.clip(fc, 45.0, 4500.0)
            
        else: # CNC Turning
            # Workpiece diameter (mm)
            wp_diam = np.random.uniform(20.0, 110.0)
            tool_diameter[i] = wp_diam
            num_flutes[i] = 1
            
            # Cutting speed (m/min)
            base_vc = np.random.uniform(80.0, 280.0) * mat_prop["speed_factor"]
            cutting_speed[i] = np.clip(base_vc, 30.0, 450.0)
            
            # Spindle Speed N = 1000 * vc / (pi * D_wp)
            n_calc = (1000.0 * cutting_speed[i]) / (np.pi * wp_diam)
            spindle_speed[i] = np.clip(n_calc, 250.0, 5000.0)
            
            # Feed rate f (mm/rev)
            f_rev = np.random.uniform(0.06, 0.40)
            feed_rate[i] = f_rev
            
            # Feed speed vf = f * N (mm/min)
            vf = f_rev * spindle_speed[i]
            feed_speed[i] = vf
            
            # Depths of cut
            axial_depth[i] = np.random.uniform(0.5, 4.0) # Depth of cut ap
            radial_depth[i] = f_rev                      # In turning, feed is radial/longitudinal advance
            
            # MRR in Turning (cm^3/min) = vc * 1000 * ap * f / 1000 = vc * ap * f
            mrr_calc = cutting_speed[i] * axial_depth[i] * f_rev
            mrr[i] = mrr_calc
            
            h = f_rev * np.sin(np.radians(75.0)) # Entering angle ~75 deg
            h = max(h, 0.01)
            
            kc11 = mat_prop["kc1_1"] * (hardness[i] / ((h_min + h_max) / 2.0)) ** 0.5
            mc = mat_prop["mc"]
            k_gamma = 1.0 - 0.014 * rake_angle[i]
            k_wear = 1.0 + 1.30 * ((tool_wear[i] / 0.30) ** 1.15)
            k_clnt = clnt_prop["force_factor"]
            k_coat = coat_prop["friction_factor"]
            
            kc = kc11 * (h ** (-mc)) * k_gamma * k_wear * k_clnt * k_coat
            fc = kc * axial_depth[i] * h
            cutting_force[i] = np.clip(fc, 50.0, 4800.0)
            
        # Machine Power Components (in kW)
        # 1. Base Controller & Auxiliaries (Hydraulics, fans, display)
        p_base_cnc = 0.50
        
        # 2. Coolant pump active power
        p_pump = clnt_prop["pump_power_kw"]
        p_aux[i] = p_base_cnc + p_pump
        
        # 3. Spindle No-load / Friction Power: P = c0 + c1*N + c2*N^2
        n_rpm = spindle_speed[i]
        p_spindle[i] = 0.45 + 0.00030 * n_rpm + 3.8e-8 * (n_rpm ** 2)
        
        # 4. Feed Axis Drive Power: P = d0 + d1*vf
        p_feed[i] = 0.15 + 0.00016 * feed_speed[i]
        
        # 5. Cutting Power: P_cut = (Fc * vc) / (60,000 * eta_motor)
        motor_efficiency = np.random.uniform(0.83, 0.89)
        p_cutting[i] = (cutting_force[i] * cutting_speed[i]) / (60000.0 * motor_efficiency)
        
        # Total Machine Active Power (kW)
        p_calc = p_aux[i] + p_spindle[i] + p_feed[i] + p_cutting[i]
        
        # Physical measurement sensor noise & high-frequency vibration
        noise_std = 0.035 * p_calc + 0.04
        p_noisy = p_calc + np.random.normal(0.0, noise_std)
        total_power[i] = np.clip(p_noisy, 0.65, 32.0)
        
        # Specific Energy Consumption (J/mm^3) = (Power in Watts) / (MRR in mm^3/s)
        # MRR (cm^3/min) = MRR * (1000 / 60) mm^3/s
        mrr_mm3_s = max(mrr[i] * 1000.0 / 60.0, 0.1)
        specific_energy[i] = (total_power[i] * 1000.0) / mrr_mm3_s

    df = pd.DataFrame({
        "Operation_Type": operations,
        "Workpiece_Material": materials,
        "Material_Hardness_HB": np.round(hardness, 1),
        "Tool_Diameter_mm": np.round(tool_diameter, 2),
        "Number_of_Flutes": num_flutes,
        "Tool_Coating": coatings,
        "Tool_Wear_VB_mm": np.round(tool_wear, 3),
        "Rake_Angle_deg": np.round(rake_angle, 1),
        "Cutting_Speed_vc_mpm": np.round(cutting_speed, 2),
        "Spindle_Speed_RPM": np.round(spindle_speed, 1),
        "Feed_Rate_mm_rev": np.round(feed_rate, 4),
        "Axial_Depth_ap_mm": np.round(axial_depth, 2),
        "Radial_Depth_ae_mm": np.round(radial_depth, 2),
        "Feed_Speed_vf_mmpm": np.round(feed_speed, 1),
        "Coolant_Condition": coolants,
        "Coolant_Flow_Rate_Lpm": np.round(coolant_flow, 2),
        "Material_Removal_Rate_cm3_min": np.round(mrr, 3),
        "Cutting_Force_Fc_N": np.round(cutting_force, 2),
        "Spindle_Power_kW": np.round(p_spindle, 3),
        "Feed_Power_kW": np.round(p_feed, 3),
        "Auxiliary_Power_kW": np.round(p_aux, 3),
        "Cutting_Power_kW": np.round(p_cutting, 3),
        "Power_Consumption_kW": np.round(total_power, 3),
        "Specific_Energy_J_mm3": np.round(specific_energy, 2)
    })
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate Machining Power Consumption Dataset")
    parser.add_argument("--samples", type=int, default=12500, help="Number of samples to generate (>10k)")
    parser.add_argument("--output", type=str, default="data/raw/machining_power_consumption_12k.csv", help="Output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print(f"Generating {args.samples} physics-grounded machining observations...")
    df = generate_machining_dataset(num_samples=args.samples, random_seed=args.seed)
    
    df.to_csv(args.output, index=False)
    print(f"Successfully generated and saved dataset to: {args.output}")
    print(f"Dataset shape: {df.shape}")
    print("\nSample preview:")
    print(df[["Operation_Type", "Workpiece_Material", "Cutting_Speed_vc_mpm", "Feed_Rate_mm_rev", "Power_Consumption_kW"]].head())
    print("\nSummary statistics of Power_Consumption_kW:")
    print(df["Power_Consumption_kW"].describe())


if __name__ == "__main__":
    main()
