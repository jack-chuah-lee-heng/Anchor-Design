import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import math

# Set page config
st.set_page_config(page_title="FOWT Load & Anchor Design", layout="wide")

# Title
st.title("Floating Offshore Wind Turbine Load Analysis & Anchor Design")
st.markdown("Based on methodologies from Arany & Bhattacharya (2018) and Nasab et al. (2022)")

# Sidebar for inputs
st.sidebar.header("Input Parameters")

# Turbine Parameters
st.sidebar.subheader("1. Wind Turbine Parameters")
D = st.sidebar.number_input("Rotor diameter (m)", min_value=50.0, max_value=250.0, value=107.0)
U_R = st.sidebar.number_input("Rated wind speed (m/s)", min_value=8.0, max_value=20.0, value=16.5)
z_hub = st.sidebar.number_input("Hub height above sea level (m)", min_value=50.0, max_value=150.0, value=87.0)
m_RNA = st.sidebar.number_input("Rotor-nacelle assembly mass (tons)", min_value=100.0, max_value=1000.0, value=300.0)
m_tower = st.sidebar.number_input("Tower mass (tons)", min_value=100.0, max_value=1000.0, value=255.0)
D_b = st.sidebar.number_input("Tower bottom diameter (m)", min_value=3.0, max_value=10.0, value=5.0)
D_t = st.sidebar.number_input("Tower top diameter (m)", min_value=2.0, max_value=8.0, value=3.0)

# Platform Parameters
st.sidebar.subheader("2. Spar Platform Parameters")
D_S = st.sidebar.number_input("Spar diameter (m)", min_value=5.0, max_value=20.0, value=14.0)
B = st.sidebar.number_input("Spar draft (m)", min_value=50.0, max_value=150.0, value=93.0)
m_B = st.sidebar.number_input("Ballast mass (tons)", min_value=1000.0, max_value=15000.0, value=8000.0)
m_S = st.sidebar.number_input("Spar mass (tons)", min_value=500.0, max_value=5000.0, value=1000.0)

# Site Conditions
st.sidebar.subheader("3. Site Conditions")
S = st.sidebar.number_input("Water depth (m)", min_value=50.0, max_value=200.0, value=110.0)
U_avg = st.sidebar.number_input("Mean wind speed (m/s)", min_value=5.0, max_value=15.0, value=7.1)
shape_param = st.sidebar.number_input("Weibull shape parameter", min_value=1.5, max_value=3.0, value=1.98)
scale_param = st.sidebar.number_input("Weibull scale parameter", min_value=5.0, max_value=15.0, value=7.99)
I_ref = st.sidebar.number_input("Reference turbulence intensity (%)", min_value=10.0, max_value=25.0, value=16.0) / 100
L_k = st.sidebar.number_input("Turbulence integral length scale (m)", min_value=200.0, max_value=500.0, value=340.2)

# Wave Parameters
st.sidebar.subheader("4. Wave Parameters")
H_S50 = st.sidebar.number_input("50-year significant wave height (m)", min_value=5.0, max_value=25.0, value=15.0)
v_c = st.sidebar.number_input("Current velocity (m/s)", min_value=0.5, max_value=5.0, value=1.32)

# Soil Parameters
st.sidebar.subheader("5. Soil Parameters")
soil_type = st.sidebar.selectbox("Soil type", ["Sand", "Clay (constant su)", "Clay (linear su)"])
if soil_type == "Sand":
    phi = st.sidebar.number_input("Internal friction angle (degrees)", min_value=25.0, max_value=40.0, value=30.0)
    gamma_prime = st.sidebar.number_input("Effective unit weight (kN/m³)", min_value=5.0, max_value=12.0, value=9.0)
elif soil_type == "Clay (constant su)":
    s_u = st.sidebar.number_input("Undrained shear strength (kPa)", min_value=10.0, max_value=100.0, value=30.0)
else:
    s_u0 = st.sidebar.number_input("Initial shear strength (kPa)", min_value=5.0, max_value=50.0, value=15.0)
    ds_u_dz = st.sidebar.number_input("Shear strength gradient (kPa/m)", min_value=0.5, max_value=5.0, value=2.0)

# Constants
rho_a = 1.225  # Air density kg/m³
rho_w = 1030   # Water density kg/m³
g = 9.81       # Gravity m/s²
C_D = 0.5      # Drag coefficient
C_m = 2.0      # Inertia coefficient
mu = 0.25      # Mooring chain friction

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.header("Load Calculations")
    
    # Wind Load Calculations
    st.subheader("Wind Load Analysis")
    
    # Calculate thrust coefficient
    C_T = 7 / U_avg
    
    # 50-year extreme wind speed
    U_10_50yr = scale_param * (-np.log(1 - 0.98**(1/52596)))**(1/shape_param)
    U_10_1yr = 0.8 * U_10_50yr
    
    # EOG calculation
    sigma_U_c = 0.11 * U_10_1yr
    Lambda_1 = L_k / 8
    u_EOG = min(1.35 * (U_10_1yr - U_R), 3.3 * sigma_U_c / (1 + 0.1 * D / Lambda_1))
    
    # Wind scenarios
    # U-1: NTM at rated
    sigma_u_NTM = I_ref * (0.75 * U_R + 5.6)
    f_1P_max = 0.218  # Hz
    sigma_u_NTM_f = sigma_u_NTM * np.sqrt(1 / ((6 * L_k / U_R) * f_1P_max + 1)**(2/3))
    u_NTM = 1.28 * sigma_u_NTM_f
    F_wind_NTM = 0.5 * rho_a * np.pi * D**2 / 4 * C_T * (U_R + u_NTM)**2 / 1e6  # MN
    
    # U-2: ETM at rated
    sigma_u_ETM = 2 * I_ref * (0.072 * (U_avg / 2 + 3) * (U_R / 2 - 4) + 10)
    sigma_u_ETM_f = sigma_u_ETM * np.sqrt(1 / ((6 * L_k / U_R) * f_1P_max + 1)**(2/3))
    u_ETM = 2 * sigma_u_ETM_f
    F_wind_ETM = 0.5 * rho_a * np.pi * D**2 / 4 * C_T * (U_R + u_ETM)**2 / 1e6  # MN
    
    # U-3: EOG at rated
    F_wind_EOG = 0.5 * rho_a * np.pi * D**2 / 4 * C_T * (U_R + u_EOG)**2 / 1e6  # MN
    
    # U-4: EOG at cut-out (25 m/s)
    U_out = 25.0
    C_T_out = 7 * U_R**2 / U_out**3
    F_wind_out = 0.5 * rho_a * np.pi * D**2 / 4 * C_T_out * (U_out + u_EOG)**2 / 1e6  # MN
    
    # Display wind loads
    wind_df = pd.DataFrame({
        'Scenario': ['U-1: NTM', 'U-2: ETM', 'U-3: EOG at rated', 'U-4: EOG at cut-out'],
        'Wind Load (MN)': [F_wind_NTM, F_wind_ETM, F_wind_EOG, F_wind_out],
        'Moment (MNm)': [F_wind_NTM * (S + z_hub), F_wind_ETM * (S + z_hub), 
                         F_wind_EOG * (S + z_hub), F_wind_out * (S + z_hub)]
    })
    st.dataframe(wind_df)
    
    # Wave Load Calculations
    st.subheader("Wave Load Analysis")
    
    # Wave parameters
    H_S1 = 0.8 * H_S50
    T_S50 = 11.1 * np.sqrt(H_S50 / g)
    T_S1 = 11.1 * np.sqrt(H_S1 / g)
    
    # Maximum wave heights
    N_50 = 10800 / T_S50
    N_1 = 10800 / T_S1
    H_m50 = H_S50 * np.sqrt(0.5 * np.log(N_50))
    H_m1 = H_S1 * np.sqrt(0.5 * np.log(N_1))
    
    # Wave periods
    T_m50 = 11.1 * np.sqrt(H_m50 / g)
    T_m1 = 11.1 * np.sqrt(H_m1 / g)
    
    # Wave number (iterative solution)
    def dispersion_relation(k, T, S):
        omega = 2 * np.pi / T
        return omega**2 - g * k * np.tanh(k * S)
    
    k_50 = fsolve(lambda k: dispersion_relation(k, T_m50, S), 0.1)[0]
    k_1 = fsolve(lambda k: dispersion_relation(k, T_m1, S), 0.1)[0]
    
    # Wave loads (simplified for W-2 and W-4)
    # W-2: 1-year EWH
    F_D_max_1yr = 0.5 * rho_w * D_S * C_D * (np.pi * H_m1 / T_m1)**2 / np.sinh(k_1 * S)**2 * B / 1e6
    F_I_max_1yr = 0.5 * rho_w * C_m * np.pi * D_S**2 / 4 * 2 * np.pi**2 * H_m1 / T_m1**2 / np.sinh(k_1 * S) * B / 1e6
    F_wave_1yr = F_D_max_1yr + F_I_max_1yr
    
    # W-4: 50-year EWH
    F_D_max_50yr = 0.5 * rho_w * D_S * C_D * (np.pi * H_m50 / T_m50)**2 / np.sinh(k_50 * S)**2 * B / 1e6
    F_I_max_50yr = 0.5 * rho_w * C_m * np.pi * D_S**2 / 4 * 2 * np.pi**2 * H_m50 / T_m50**2 / np.sinh(k_50 * S) * B / 1e6
    F_wave_50yr = F_D_max_50yr + F_I_max_50yr
    
    # Current load
    F_C = 0.5 * rho_w * D_S * C_D * v_c**2 * B / 1e6  # MN
    
    wave_df = pd.DataFrame({
        'Scenario': ['W-2: 1-year EWH', 'W-4: 50-year EWH', 'Current'],
        'Load (MN)': [F_wave_1yr, F_wave_50yr, F_C]
    })
    st.dataframe(wave_df)
    
    # Load Combinations
    st.subheader("ULS Load Combinations")
    
    gamma_L = 1.35  # Load factor
    
    # E-1: Normal operation
    F_E1 = F_wind_NTM + F_wave_1yr + F_C
    # E-2: Extreme wave
    F_E2 = F_wind_ETM + F_wave_50yr + F_C
    # E-3: Extreme wind
    F_E3 = F_wind_EOG + F_wave_1yr + F_C
    # E-4: Cut-out
    F_E4 = F_wind_out + F_wave_50yr + F_C
    # E-5: Misaligned
    F_E5 = F_E2  # Same as E-2 for horizontal load
    
    combo_df = pd.DataFrame({
        'Load Case': ['E-1', 'E-2', 'E-3', 'E-4', 'E-5'],
        'Total Load (MN)': [F_E1, F_E2, F_E3, F_E4, F_E5],
        'Factored Load (MN)': [F_E1 * gamma_L, F_E2 * gamma_L, F_E3 * gamma_L, 
                               F_E4 * gamma_L, F_E5 * gamma_L]
    })
    st.dataframe(combo_df)
    
    # Design load
    F_design = max(F_E1, F_E2, F_E3, F_E4, F_E5) * gamma_L
    st.metric("Design Load", f"{F_design:.2f} MN")

with col2:
    st.header("Anchor Design")
    
    # Anchor sizing
    st.subheader("Suction Caisson Design")
    
    L_D_ratio = st.slider("Length to diameter ratio (L/D)", min_value=2.0, max_value=5.0, value=3.2, step=0.1)
    
    if soil_type == "Sand":
        # Sand calculations
        N_q = np.exp(np.pi * np.tan(np.radians(phi))) * np.tan(np.radians(45 + phi/2))**2
        
        # Initial guess for diameter
        D_caisson = 7.0
        L_caisson = D_caisson * L_D_ratio
        
        # Horizontal capacity
        H_m = 0.5 * D_caisson * N_q * gamma_prime * L_caisson**2 / 1000  # MN
        
        # Vertical capacity
        W_caisson = np.pi * D_caisson * L_caisson * 0.1 * 7.86 * g / 1000  # MN (assuming t/D = 0.1)
        K_tan_delta_e = 7.0
        K_tan_delta_i = 5.0
        Z_e = D_caisson / (4 * K_tan_delta_e)
        Z_i = (D_caisson - 0.2) / (4 * K_tan_delta_i)
        
        def y_func(x):
            return np.exp(-x) - 1 + x
        
        V_ext = gamma_prime * Z_e**2 * y_func(L_caisson/Z_e) * K_tan_delta_e * np.pi * D_caisson / 1000
        V_int = gamma_prime * Z_i**2 * y_func(L_caisson/Z_i) * K_tan_delta_i * np.pi * (D_caisson - 0.2) / 1000
        V_m = W_caisson + V_ext + V_int
        
    else:
        # Clay calculations
        if soil_type == "Clay (constant su)":
            s_u_avg = s_u
            s_u_tip = s_u
        else:
            s_u_avg = s_u0 + ds_u_dz * L_D_ratio * 3.5  # Approximate
            s_u_tip = s_u0 + ds_u_dz * L_D_ratio * 7
        
        D_caisson = 5.0
        L_caisson = D_caisson * L_D_ratio
        
        # Horizontal capacity
        N_p = 10.0  # Lateral bearing capacity factor
        H_m = L_caisson * D_caisson * N_p * s_u_avg / 1000  # MN
        
        # Vertical capacity
        W_caisson = np.pi * D_caisson * L_caisson * 0.07 * 7.86 * g / 1000  # MN
        alpha_e = 0.5
        alpha_i = 0.4
        N_c = 9.0
        
        A_se = np.pi * D_caisson * L_caisson
        A_si = np.pi * (D_caisson - 0.14) * L_caisson
        A_e = np.pi * D_caisson**2 / 4
        
        V_m1 = W_caisson + A_se * alpha_e * s_u_avg / 1000 + N_c * s_u_tip * A_e / 1000
        V_m2 = W_caisson + A_se * alpha_e * s_u_avg / 1000 + A_si * alpha_i * s_u_avg / 1000
        V_m3 = W_caisson + A_se * alpha_e * s_u_avg / 1000 + np.pi * (D_caisson - 0.14)**2 / 4 * L_caisson * gamma_prime / 1000
        V_m = min(V_m1, V_m2, V_m3)
    
    # Anchor padeye location
    if soil_type == "Sand":
        z_a = 2/3 * L_caisson
    else:
        z_a = 0.5 * L_caisson
    
    # Load at anchor (simplified)
    theta_m = np.radians(30)  # Assumed mudline angle
    T_m = F_design
    
    # Solve for anchor loads
    def anchor_equations(vars):
        T_a, theta_a = vars
        eq1 = T_a**2 * (theta_a**2 - theta_m**2) - z_a * 0.5 * D_caisson * N_q * gamma_prime * z_a / 1000
        eq2 = T_m / T_a - np.exp(mu * (theta_a - theta_m))
        return [eq1, eq2]
    
    try:
        T_a, theta_a = fsolve(anchor_equations, [F_design * 1.1, 0.5])
        H_u = T_a * np.cos(theta_a)
        V_u = T_a * np.sin(theta_a)
    except:
        H_u = F_design * 0.9
        V_u = F_design * 0.4
    
    # Check capacity
    a = L_D_ratio + 0.5
    b = L_D_ratio / 3 + 4.5
    
    F_P = (H_u / H_m)**a + (V_u / V_m)**b
    
    # Display results
    anchor_results = {
        'Parameter': ['Caisson diameter', 'Caisson length', 'Horizontal capacity', 
                      'Vertical capacity', 'Horizontal load', 'Vertical load', 
                      'Capacity ratio (FP)'],
        'Value': [f"{D_caisson:.2f} m", f"{L_caisson:.2f} m", f"{H_m:.2f} MN", 
                  f"{V_m:.2f} MN", f"{H_u:.2f} MN", f"{V_u:.2f} MN", f"{F_P:.2f}"],
        'Status': ['', '', '', '', '', '', 
                   '✓ OK' if F_P < 1.0 else '✗ Resize needed']
    }
    
    results_df = pd.DataFrame(anchor_results)
    st.dataframe(results_df)
    
    # Optimization suggestion
    if F_P > 1.0:
        suggested_D = D_caisson * (F_P ** (1/3))
        st.warning(f"⚠️ Anchor capacity exceeded! Suggested diameter: {suggested_D:.2f} m")
    else:
        st.success(f"✓ Anchor design is adequate with safety factor: {1/F_P:.2f}")
    
    # Visualization
    st.subheader("Load Distribution")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Wind vs Wave loads
    scenarios = ['E-1', 'E-2', 'E-3', 'E-4']
    wind_loads = [F_wind_NTM, F_wind_ETM, F_wind_EOG, F_wind_out]
    wave_loads = [F_wave_1yr, F_wave_50yr, F_wave_1yr, F_wave_50yr]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    ax1.bar(x - width/2, wind_loads, width, label='Wind', color='skyblue')
    ax1.bar(x + width/2, wave_loads, width, label='Wave', color='navy')
    ax1.set_xlabel('Load Case')
    ax1.set_ylabel('Load (MN)')
    ax1.set_title('Wind vs Wave Loads')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Capacity utilization
    labels = ['Horizontal', 'Vertical']
    utilization = [H_u/H_m * 100, V_u/V_m * 100]
    colors = ['green' if u < 100 else 'red' for u in utilization]
    
    ax2.bar(labels, utilization, color=colors, alpha=0.7)
    ax2.axhline(y=100, color='red', linestyle='--', label='Capacity limit')
    ax2.set_ylabel('Utilization (%)')
    ax2.set_title('Anchor Capacity Utilization')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

# Export results
st.header("Export Results")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Generate Report"):
        report = f"""
FLOATING WIND TURBINE LOAD ANALYSIS REPORT
==========================================

PROJECT PARAMETERS:
- Turbine: {D}m rotor, {U_R} m/s rated
- Platform: {D_S}m x {B}m spar
- Site: {S}m water depth, {U_avg} m/s mean wind
- Soil: {soil_type}

LOAD SUMMARY:
- Design Load: {F_design:.2f} MN
- Critical Case: E-{np.argmax([F_E1, F_E2, F_E3, F_E4, F_E5]) + 1}

ANCHOR DESIGN:
- Type: Suction Caisson
- Diameter: {D_caisson:.2f} m
- Length: {L_caisson:.2f} m
- Capacity Ratio: {F_P:.2f}
- Status: {'ADEQUATE' if F_P < 1.0 else 'RESIZE REQUIRED'}
"""
        st.text_area("Report", report, height=400)

with col2:
    if st.button("Save Parameters"):
        params = {
            'turbine': {'D': D, 'U_R': U_R, 'z_hub': z_hub},
            'platform': {'D_S': D_S, 'B': B, 'm_B': m_B},
            'site': {'S': S, 'U_avg': U_avg, 'H_S50': H_S50},
            'loads': {'F_design': F_design},
            'anchor': {'D': D_caisson, 'L': L_caisson, 'F_P': F_P}
        }
        st.json(params)

with col3:
    st.info("This tool implements simplified methods from Arany & Bhattacharya (2018) and Nasab et al. (2022) for preliminary design.")

# Footer
st.markdown("---")
st.markdown("Developed for parametric design of floating offshore wind turbine foundations")
