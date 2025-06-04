import streamlit as st
import math
import numpy as np
from scipy.optimize import fsolve

# --- Constants ---
G_ACCEL = 9.81
RHO_AIR_DEFAULT = 1.225  # kg/m^3
RHO_WATER_DEFAULT = 1025.0 # kg/m^3 (can be adjusted for seawater) - Changed to float
PI = math.pi

# --- Helper Functions ---
def to_radians(degrees):
    return degrees * PI / 180

def to_degrees(radians):
    return radians * 180 / PI

# --- Wind Load Calculations (Ref: AB2018 Sec 2.1.1, 2.1.2; MN2022 Sec 5.1) ---
def calculate_extreme_wind_speeds(weibull_lambda, weibull_k, N_intervals_year=52596):
    # AB2018 Eq 5 (adapted for U10,50yr); MN2022 Eq 11
    if weibull_lambda <= 0 or weibull_k <= 0: return 0, 0
    try:
        U10_50yr = weibull_lambda * (-math.log(1 - 0.98**(1/N_intervals_year)))**(1/weibull_k)
    except ValueError: # log of negative if 0.98**(1/N_intervals_year) is 1 or more
        U10_50yr = 0
    # AB2018 (implicitly); MN2022 Eq 12
    U10_1yr = 0.8 * U10_50yr
    return U10_50yr, U10_1yr

def calculate_sigma_Uc(U10_1yr):
    # AB2018 (part of EOG); MN2022 Eq 13
    return 0.11 * U10_1yr

def calculate_eog_speed(U10_1yr, U_R_turbine, sigma_Uc, Lk_turbulence, D_rotor):
    # AB2018 Eq 7; MN2022 Eq 14
    # Note: AB2018 uses Lambda_1 = Lk/8.1, MN2022 uses Lambda_1 = Lk/8. Using Lk/8.1
    if Lk_turbulence <=0: return 0
    Lambda_1 = Lk_turbulence / 8.1 # Turbulence scale parameter
    term1 = 1.35 * (U10_1yr - U_R_turbine) if U10_1yr > U_R_turbine else float('inf')
    term2 = (3.3 * sigma_Uc) / (1 + (0.1 * D_rotor) / Lambda_1) if Lambda_1 > 0 else float('inf') # D_rotor not A1 in AB2018, using D_rotor as characteristic length
    return min(term1, term2)

def get_thrust_coefficient(U_wind, U_R_turbine, C_T_method="frohboese_ab2018", U_cut_in=4.0, U_cut_out=25.0):
    # AB2018 Eq 2 (Frohboese & Schmuck 2010 based)
    # MN2022 Eq 16 (u_bar = U_wind), Eq 18
    if U_wind < U_cut_in or U_wind > U_cut_out : return 0.2 # Approx for parked/idling
    
    if C_T_method == "frohboese_ab2018":
        if U_R_turbine <= 0: return 0.8 # Approx peak
        # This formula is for C_T around rated speed.
        # For a full thrust curve, more complex logic or a lookup table is needed.
        # Using a simplified approach for this app: peak C_T up to U_R, then decreases.
        if U_wind <= U_R_turbine:
             # Approximation: linear increase to max C_T (e.g. 0.8-0.9) then constant
             # Or use AB2018 Eq2 if wind is close to U_R
             # For simplicity, let's use a typical max C_T for FOWTs
             return 0.85 # Typical peak thrust coefficient
        else: # Above rated speed
            # Power = 0.5 * rho * A * C_P * U^3 = constant
            # Thrust = 0.5 * rho * A * C_T * U^2
            # If C_P is constant, C_T ~ 1/U. If Power is constant, C_T ~ 1/U^3 * P_rated / (0.5*rho*A)
            # Using MN2022 Eq 18 logic: C_T = (7 * U_R_turbine**2) / U_wind**3 (7 is empirical)
            # This can lead to very high C_T if U_wind is small and > U_R.
            # Let's use a simpler 1/U^2 relationship for thrust reduction past rated for constant power.
            # F_thrust_rated * U_R = F_thrust_U * U => C_T_U * U^3 = C_T_R * U_R^3
            C_T_rated = 0.85
            return C_T_rated * (U_R_turbine / U_wind)**2 if U_wind > 0 else 0 # Simplified, assumes constant power output by varying pitch
            
    elif C_T_method == "simplified_mn2022":
        if U_wind <= U_R_turbine:
            return 7.0 / U_wind if U_wind > 0 else 0 # MN2022 Eq 16 (u_bar = U_wind)
        else:
            return (7.0 * U_R_turbine**2) / U_wind**3 if U_wind > 0 else 0 # MN2022 Eq 18
    return 0.8 # Default fallback

def calculate_rotor_thrust_force(rho_air, D_rotor, C_T_thrust, U_effective_wind):
    A_rotor = PI * (D_rotor/2)**2
    return 0.5 * rho_air * A_rotor * C_T_thrust * U_effective_wind**2

def calculate_tower_drag_force(rho_air, C_DT_tower, U_hub_wind, z_hub, D_tower_bottom, D_tower_top, gamma_shear):
    # AB2018 Eq 11
    if z_hub <= 0 or (2*gamma_shear + 1) == 0 or (2*gamma_shear + 2) == 0: return 0
    # Integral of D(z)*U(z)^2. U(z) = U_hub * (z/z_hub)^gamma_shear
    # D(z) = D_b - (D_b - D_t)/z_hub * z
    # Simplified using the direct formula from AB2018 Eq 11:
    factor_diam = (D_tower_bottom + (2*gamma_shear + 1)*D_tower_top) / ((2*gamma_shear + 1)*(2*gamma_shear + 2))
    F_DT = 0.5 * rho_air * C_DT_tower * U_hub_wind**2 * z_hub * factor_diam
    return F_DT

def calculate_shutdown_thrust_drag(rho_air, U10_50yr, D_rotor, C_D_rotor_parked, F_DT_at_U50):
    # Simplified version of AB2018 Eq 8.
    # Assumes a drag coefficient for the entire parked rotor.
    A_rotor = PI * (D_rotor/2)**2
    F_rotor_drag = 0.5 * rho_air * A_rotor * C_D_rotor_parked * U10_50yr**2
    return F_rotor_drag + F_DT_at_U50


# --- Wave Load Calculations (Ref: AB2018 Sec 2.1.3; MN2022 Sec 5.2) ---
def get_wave_design_parameters(H_S50_known, g=G_ACCEL):
    # AB2018 Eq 12, 13, 14; MN2022 Eq 36-40
    H_S1 = 0.8 * H_S50_known
    
    T_S50 = 11.1 * math.sqrt(H_S50_known / g) if H_S50_known > 0 and g > 0 else 0.0
    T_S1 = 11.1 * math.sqrt(H_S1 / g) if H_S1 > 0 and g > 0 else 0.0
    
    N_waves_50 = (3 * 3600) / T_S50 if T_S50 > 0 else 0.0
    N_waves_1 = (3 * 3600) / T_S1 if T_S1 > 0 else 0.0
    
    H_m50 = H_S50_known * math.sqrt(0.5 * math.log(N_waves_50)) if N_waves_50 > 1 and H_S50_known > 0 else 0.0 # log(N) requires N > 1
    H_m1 = H_S1 * math.sqrt(0.5 * math.log(N_waves_1)) if N_waves_1 > 1 and H_S1 > 0 else 0.0
    
    T_m50 = 11.1 * math.sqrt(H_m50 / g) if H_m50 > 0 and g > 0 else 0.0
    T_m1 = 11.1 * math.sqrt(H_m1 / g) if H_m1 > 0 and g > 0 else 0.0
    
    return H_S1, T_S1, H_m1, T_m1, H_S50_known, T_S50, H_m50, T_m50

def dispersion_equation(k, omega, S_water_depth, g=G_ACCEL):
    if k <= 0: k = 1e-6 # k must be positive, avoid issues with tanh(0) or negative k
    try:
        tanh_val = math.tanh(k * S_water_depth)
    except OverflowError:
        tanh_val = 1.0 # for very large k*S_water_depth
    return omega**2 - g * k * tanh_val

def solve_dispersion_relation_iterative(omega_wave, S_water_depth, g=G_ACCEL):
    if omega_wave == 0: return 0.0
    # Initial guess for k (deep water approximation: k_approx = omega^2 / g)
    k_initial_guess = omega_wave**2 / g
    if k_initial_guess <= 0: k_initial_guess = 1e-6

    try:
        # Check if S_water_depth is very small, might cause issues with tanh if k is also small
        if S_water_depth < 1e-3: # effectively shallow water for any practical omega
             if g * S_water_depth > 0:
                return omega_wave / math.sqrt(g * S_water_depth)
             else:
                return 1e-6

        k_solution, = fsolve(dispersion_equation, k_initial_guess, args=(omega_wave, S_water_depth, g), xtol=1e-8, maxfev=500)
        return abs(k_solution) if k_solution != 0 else 1e-6 # Ensure positive k, avoid zero
    except Exception as e:
        # Fallback for shallow water: k_approx = omega / sqrt(g*S)
        # st.warning(f"Dispersion solver failed: {e}. Using shallow water approx.")
        if S_water_depth > 0 and g * S_water_depth > 0:
            k_fallback = omega_wave / math.sqrt(g * S_water_depth)
            return abs(k_fallback) if k_fallback != 0 else 1e-6
        return 1e-6


def calculate_morison_wave_forces(rho_water, D_spar, C_D_spar, C_m_spar, H_m_wave, T_m_wave, S_water_depth, B_spar_draft, k_wave_number):
    # AB2018 Eq 20, 21 (integrated forms P_D, P_I)
    if T_m_wave == 0 or k_wave_number == 0: return 0.0, 0.0
    
    try:
        sinh_kS_val = math.sinh(k_wave_number * S_water_depth)
        if sinh_kS_val == 0: return 0.0, 0.0
    except OverflowError: # kS is too large
        return 0.0, 0.0 # Effectively infinite denominator or invalid scenario

    # Max Drag Force (eta = H_m/2 at t=0 for u(z,0))
    # P_D from AB2018 Eq 20
    try:
        term_exp1 = math.exp(2 * k_wave_number * (S_water_depth + H_m_wave / 2))
        term_exp2 = math.exp(-2 * k_wave_number * (S_water_depth + H_m_wave / 2))
        term_exp3 = math.exp(2 * k_wave_number * (S_water_depth - B_spar_draft))
        term_exp4 = math.exp(-2 * k_wave_number * (S_water_depth - B_spar_draft))
    except OverflowError:
        return 0.0, 0.0 # exp arguments too large

    P_D = (1 / (8 * k_wave_number)) * (term_exp1 - term_exp2 - term_exp3 + term_exp4) + H_m_wave / 4 + B_spar_draft / 2

    try:
        sinh_kS_sq = sinh_kS_val**2
        if sinh_kS_sq == 0: return 0.0, 0.0
    except OverflowError:
         return 0.0, 0.0

    F_D_max = 0.5 * rho_water * D_spar * C_D_spar * (PI**2 * H_m_wave**2) / (T_m_wave**2 * sinh_kS_sq) * P_D

    # Max Inertia Force (eta = 0 at t=Tm/4 for u_dot(z, Tm/4))
    # P_I from AB2018 Eq 21
    try:
        sinh_k_S_minus_B = math.sinh(k_wave_number * (S_water_depth - B_spar_draft))
    except OverflowError:
        sinh_k_S_minus_B = float('inf') if k_wave_number * (S_water_depth - B_spar_draft) > 0 else float('-inf')


    P_I = (sinh_kS_val - sinh_k_S_minus_B) / k_wave_number
    A_p_spar = PI * (D_spar/2)**2
    F_I_max = C_m_spar * rho_water * A_p_spar * (2 * PI**2 * H_m_wave) / (T_m_wave**2 * sinh_kS_val) * P_I
    
    return F_D_max, F_I_max

# --- Current Load Calculation (Ref: AB2018 Sec 2.1.4; MN2022 Eq 41) ---
def calculate_current_force(rho_water, D_spar, C_D_current_spar, v_current_speed, B_spar_draft):
    # AB2018 Eq 25
    return 0.5 * rho_water * D_spar * C_D_current_spar * v_current_speed**2 * B_spar_draft

# --- Load Transfer to Anchor (Ref: AB2018 Sec 3.4.3; MN2022 Sec 5.4) ---
def calculate_Q_avg_forerunner(soil_type, s_u_avg_clay_Pa, gamma_prime_sand_Nm3, z_a_padeye, A_b_forerunner, Nc_forerunner_bearing):
    # AB2018 Eq 52 (clay), 53 (sand)
    # s_u_avg_clay_Pa in Pa (N/m^2)
    # gamma_prime_sand_Nm3 in N/m^3
    if soil_type == "Clay":
        # Q_av is average soil resistance PER UNIT LENGTH of forerunner (N/m).
        return A_b_forerunner * Nc_forerunner_bearing * s_u_avg_clay_Pa
    elif soil_type == "Sand":
        # z_a * Q_av = A_b * Nc * integral(gamma_prime*z dz) = A_b * Nc * gamma_prime * z_a^2 / 2
        # Q_av = A_b * Nc * gamma_prime * z_a / 2
        return A_b_forerunner * Nc_forerunner_bearing * gamma_prime_sand_Nm3 * z_a_padeye / 2.0
    return 0.0

def anchor_load_equations(vars_in, T_mudline, theta_mudline_rad, z_a_padeye, Q_avg, mu_soil_friction):
    T_a, theta_a_rad = vars_in
    if T_a <= 1e-6: T_a = 1e-6 # Avoid division by zero or log of non-positive, ensure small positive

    # AB2018 Eq 50, 51 (modified for direct solving)
    # Eq 50: T_a/2 * (theta_a^2 - theta_m^2) - z_a * Q_av = 0
    # Eq 51: T_m / T_a - exp(mu * (theta_a - theta_m)) = 0  => T_m - T_a * exp(mu * (theta_a - theta_m)) = 0
    eq1 = (T_a / 2.0) * (theta_a_rad**2 - theta_mudline_rad**2) - z_a_padeye * Q_avg
    
    try:
        exp_term = math.exp(mu_soil_friction * (theta_a_rad - theta_mudline_rad))
    except OverflowError:
        exp_term = float('inf') if mu_soil_friction * (theta_a_rad - theta_mudline_rad) > 0 else 0.0
    eq2 = T_mudline - T_a * exp_term
    return (eq1, eq2)

def solve_anchor_padeye_load_system(T_mudline, theta_mudline_deg, z_a_padeye, Q_avg, mu_soil_friction):
    theta_mudline_rad = to_radians(theta_mudline_deg)
    # Initial guesses: T_a slightly less than T_m, theta_a slightly more than theta_m
    initial_guesses = [T_mudline * 0.95 if T_mudline > 0 else 1e-3, theta_mudline_rad + to_radians(1.0)]
    
    if Q_avg <=0 or z_a_padeye <=0 or T_mudline <=0 : # No soil resistance or no load, T_a = T_m, theta_a = theta_m
        return T_mudline, theta_mudline_deg

    try:
        solution = fsolve(anchor_load_equations, initial_guesses, args=(T_mudline, theta_mudline_rad, z_a_padeye, Q_avg, mu_soil_friction), xtol=1e-6, maxfev=500)
        T_a_sol, theta_a_rad_sol = solution
        if T_a_sol < 0 : T_a_sol = 0 # Physical constraint
        return T_a_sol, to_degrees(theta_a_rad_sol)
    except Exception as e:
        # st.warning(f"Solver for anchor padeye load failed: {e}. Returning mudline values.")
        return T_mudline, theta_mudline_deg


# --- Suction Caisson Sizing (Ref: AB2018 Sec 3.4) ---
def estimate_caisson_weights(D_caisson, L_caisson, t_wall_D_ratio, rho_steel_submerged_kg_m3, gamma_prime_soil_plug_Nm3, soil_type="Clay"):
    # rho_steel_submerged_kg_m3: submerged density of steel (kg/m3)
    # gamma_prime_soil_plug_Nm3: submerged unit weight of soil plug (N/m3)
    
    t_wall = D_caisson / t_wall_D_ratio if t_wall_D_ratio > 0 else 0.05 # Default 5cm if ratio is bad
    if t_wall * 2 >= D_caisson: t_wall = D_caisson / 4 # Ensure D_i is positive
    D_i_caisson = D_caisson - 2 * t_wall
    
    Volume_steel = PI * ( (D_caisson/2.0)**2 - (D_i_caisson/2.0)**2 ) * L_caisson
    W_prime_caisson_N = Volume_steel * rho_steel_submerged_kg_m3 * G_ACCEL # Submerged weight in N
    
    W_prime_plug_N = 0.0
    if soil_type == "Clay" or soil_type == "Sand": # Soil plug relevant for both
        Area_plug = PI * (D_i_caisson/2.0)**2
        W_prime_plug_N = Area_plug * L_caisson * gamma_prime_soil_plug_Nm3
        
    return W_prime_caisson_N, W_prime_plug_N, t_wall, D_i_caisson

def calculate_Nq_sand(phi_sand_degrees):
    # AB2018 Eq 48
    phi_rad = to_radians(phi_sand_degrees)
    if math.cos(phi_rad) == 0 or math.cos(PI/4 + phi_rad/2) == 0: return float('inf') # Avoid division by zero in tan
    try:
        return math.exp(PI * math.tan(phi_rad)) * (math.tan(PI/4.0 + phi_rad/2.0)**2)
    except OverflowError:
        return float('inf')


def calculate_caisson_capacity_clay(L_caisson, D_e_caisson, D_i_caisson, avg_s_u_kPa, tip_s_u_kPa, N_p_lateral_clay, Nc_tip_clay, alpha_e_clay, alpha_i_clay, W_prime_caisson_N, W_prime_plug_N):
    # AB2018 Sec 3.4.1
    # avg_s_u_kPa, tip_s_u_kPa in kPa
    avg_s_u_Pa = avg_s_u_kPa * 1000.0
    tip_s_u_Pa = tip_s_u_kPa * 1000.0

    # Horizontal Capacity Hm (Eq 42)
    H_m_N = L_caisson * D_e_caisson * N_p_lateral_clay * avg_s_u_Pa
    
    # Vertical Capacities Vm1, Vm2, Vm3
    A_se_external_shaft_area = PI * D_e_caisson * L_caisson
    A_si_internal_shaft_area = PI * D_i_caisson * L_caisson
    A_e_external_cross_section_area = PI * (D_e_caisson/2.0)**2
    
    F_ext_friction_N = A_se_external_shaft_area * alpha_e_clay * avg_s_u_Pa
    F_int_friction_N = A_si_internal_shaft_area * alpha_i_clay * avg_s_u_Pa # Assuming avg_s_u for internal too
    F_rev_end_bearing_N = Nc_tip_clay * tip_s_u_Pa * A_e_external_cross_section_area # tip_s_u at caisson tip
    
    V_m1_N = W_prime_caisson_N + F_ext_friction_N + F_rev_end_bearing_N # Passive suction + rev end bearing
    V_m2_N = W_prime_caisson_N + F_ext_friction_N + F_int_friction_N    # No passive suction, caisson pullout (internal friction)
    V_m3_N = W_prime_caisson_N + F_ext_friction_N + W_prime_plug_N      # No passive suction, plug pullout
    
    V_m_N = min(V_m1_N, V_m2_N, V_m3_N) if W_prime_caisson_N >= 0 else 0 # Ensure physical if caisson weight is an issue
    if V_m_N < 0 : V_m_N = 0 # Capacity cannot be negative

    return H_m_N, V_m_N, F_ext_friction_N, F_int_friction_N, F_rev_end_bearing_N, V_m1_N, V_m2_N, V_m3_N

def y_x_sand_capacity(x_val):
    # AB2018 Eq 49 related term; MN2022 Eq 45 related term y(x) = exp(-x) - 1 + x
    try:
        return math.exp(-x_val) - 1.0 + x_val
    except OverflowError:
        return float('inf') # If exp(-x) underflows to 0, then x-1. If x is large, this can be large.


def calculate_caisson_capacity_sand(L_caisson, D_e_caisson, D_i_caisson, gamma_prime_sand_Nm3, N_q_sand, W_prime_caisson_N, K_tan_delta_ext, K_tan_delta_int):
    # AB2018 Sec 3.4.2
    # gamma_prime_sand_Nm3 in N/m^3
    # Horizontal Capacity Hm (Eq 46 interpretation)
    H_m_N = 0.5 * D_e_caisson * N_q_sand * gamma_prime_sand_Nm3 * L_caisson**2
    
    # Vertical Capacity Vm (Eq 49)
    if K_tan_delta_ext == 0 or K_tan_delta_int == 0 : return H_m_N, 0.0 # Avoid division by zero
    
    Z_e = D_e_caisson / (4.0 * K_tan_delta_ext) if K_tan_delta_ext != 0 else float('inf')
    Z_i = D_i_caisson / (4.0 * K_tan_delta_int) if K_tan_delta_int != 0 else float('inf')
    
    term_ext_friction_N = 0.0
    if Z_e > 0 and Z_e != float('inf'):
        h_div_Z_e = L_caisson / Z_e
        term_ext_friction_N = gamma_prime_sand_Nm3 * Z_e**2 * y_x_sand_capacity(h_div_Z_e) * K_tan_delta_ext * PI * D_e_caisson
        
    term_int_friction_N = 0.0
    if Z_i > 0 and Z_i != float('inf'):
        h_div_Z_i = L_caisson / Z_i
        term_int_friction_N = gamma_prime_sand_Nm3 * Z_i**2 * y_x_sand_capacity(h_div_Z_i) * K_tan_delta_int * PI * D_i_caisson
        
    V_m_N = W_prime_caisson_N + term_ext_friction_N + term_int_friction_N
    if V_m_N < 0: V_m_N = 0.0 # Capacity cannot be negative
    return H_m_N, V_m_N


def check_caisson_failure_criterion(H_u_on_anchor_N, V_u_on_anchor_N, H_m_capacity_N, V_m_capacity_N, L_caisson, D_caisson):
    # AB2018 Eq 40, 41
    if D_caisson == 0 or V_m_capacity_N <= 1e-9 or H_m_capacity_N <= 1e-9: return float('inf'), 0.0, 0.0 # Avoid division by zero or log of zero
    
    a_exp = (L_caisson / D_caisson) + 0.5
    b_exp = (L_caisson / (3.0 * D_caisson)) + 4.5
    
    term_H = H_u_on_anchor_N / H_m_capacity_N if H_m_capacity_N > 0 else float('inf')
    term_V = V_u_on_anchor_N / V_m_capacity_N if V_m_capacity_N > 0 else float('inf')

    # Ensure non-negative bases for exponentiation if loads can be negative (though V_u is usually positive for uplift)
    term_H = max(0, term_H)
    term_V = max(0, term_V)
    
    try:
        FP = (term_H)**a_exp + (term_V)**b_exp
    except (ValueError, OverflowError): # e.g. large base to large power, or 0^negative
        FP = float('inf')
    return FP, a_exp, b_exp

# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("⚓ Floating Wind Turbine Anchor Design Tool")
st.markdown("""
This application performs simplified load estimation and suction caisson anchor sizing for floating offshore wind turbines,
based on methodologies from Arany & Bhattacharya (2018) and Majdi Nasab et al. (2022).
Enter parameters in the sidebar and click "Run Analysis".
""")

# --- Sidebar Inputs ---
st.sidebar.header("Input Parameters")

# Project Info
st.sidebar.subheader("Project Information")
project_name = st.sidebar.text_input("Project Name", "Demo Project")

# Turbine Parameters
st.sidebar.subheader("Turbine Parameters")
U_R_turbine = st.sidebar.number_input("Rated Wind Speed (U_R, m/s)", min_value=5.0, value=12.0, step=0.5, format="%.1f")
D_rotor = st.sidebar.number_input("Rotor Diameter (D_rotor, m)", min_value=50.0, value=154.0, step=1.0, format="%.1f")
z_hub = st.sidebar.number_input("Hub Height above MSL (z_hub, m)", min_value=50.0, value=100.0, step=1.0, format="%.1f")
D_tower_top = st.sidebar.number_input("Tower Top Diameter (D_t, m)", min_value=1.0, value=4.1, step=0.1, format="%.1f")
D_tower_bottom = st.sidebar.number_input("Tower Bottom Diameter (D_b, m)", min_value=1.0, value=6.5, step=0.1, format="%.1f")
U_cut_in_turbine = st.sidebar.number_input("Cut-in Wind Speed (m/s)", min_value=1.0, value=4.0, step=0.5)
U_cut_out_turbine = st.sidebar.number_input("Cut-out Wind Speed (m/s)", min_value=15.0, value=25.0, step=0.5)
C_T_method_choice = st.sidebar.selectbox("Thrust Coefficient (C_T) Method", ["frohboese_ab2018", "simplified_mn2022"], index=0)
rho_air = st.sidebar.number_input("Air Density (ρ_air, kg/m³)", min_value=1.0, value=RHO_AIR_DEFAULT, step=0.005, format="%.3f")
C_DT_tower = st.sidebar.number_input("Tower Drag Coefficient (C_DT)", min_value=0.1, value=0.5, step=0.05, format="%.2f")
C_D_rotor_parked = st.sidebar.number_input("Parked Rotor Drag Coefficient (C_D_rotor_parked)", min_value=0.1, value=0.8, step=0.05, format="%.2f")


# Site Parameters
st.sidebar.subheader("Site Parameters")
S_water_depth = st.sidebar.number_input("Water Depth (S, m)", min_value=10.0, value=100.0, step=5.0, format="%.1f")
# Wind
weibull_lambda = st.sidebar.number_input("Weibull Scale Parameter (λ or K, m/s)", min_value=1.0, value=8.0, step=0.1, format="%.1f")
weibull_k_shape = st.sidebar.number_input("Weibull Shape Parameter (k or s)", min_value=1.0, value=1.8, step=0.1, format="%.1f")
Lk_turbulence = st.sidebar.number_input("Integral Turbulence Length Scale (L_k, m)", min_value=50.0, value=340.2, step=10.0, format="%.1f")
gamma_shear = st.sidebar.number_input("Wind Shear Exponent (γ)", min_value=0.05, value=float(1/7), step=0.01, format="%.3f") # Ensure float
# Wave
H_S50_known = st.sidebar.number_input("50-year Significant Wave Height (H_S50, m)", min_value=1.0, value=10.0, step=0.5, format="%.1f")
rho_water = st.sidebar.number_input("Sea Water Density (ρ_water, kg/m³)", min_value=1000.0, value=RHO_WATER_DEFAULT, step=1.0, format="%.1f")
# Current
v_current_speed = st.sidebar.number_input("Mean Current Speed (v_c, m/s)", min_value=0.0, value=1.5, step=0.1, format="%.1f")

# Spar/Platform Parameters
st.sidebar.subheader("Spar/Platform Parameters")
D_spar = st.sidebar.number_input("Spar Diameter (D_P or D_S, m)", min_value=1.0, value=14.4, step=0.1, format="%.1f")
B_spar_draft = st.sidebar.number_input("Spar Draft (B, m)", min_value=10.0, value=85.0, step=1.0, format="%.1f")
C_D_spar_wave = st.sidebar.number_input("Spar Drag Coeff. for Waves (C_D_wave)", min_value=0.1, value=0.6, step=0.05, format="%.2f")
C_m_spar_wave = st.sidebar.number_input("Spar Inertia Coeff. for Waves (C_m_wave)", min_value=0.5, value=2.0, step=0.1, format="%.1f")
C_D_spar_current = st.sidebar.number_input("Spar Drag Coeff. for Current (C_D_current)", min_value=0.1, value=0.7, step=0.05, format="%.2f")

# Mooring & Anchor Parameters
st.sidebar.subheader("Mooring & Anchor Parameters")
soil_type = st.sidebar.selectbox("Soil Type", ["Clay", "Sand"], index=0)
# Forerunner
A_b_forerunner = st.sidebar.number_input("Forerunner Effective Bearing Area (A_b, m²)", min_value=0.01, value=0.3, step=0.01, format="%.2f")
Nc_forerunner_bearing = st.sidebar.number_input("Forerunner Bearing Capacity Factor (N_c)", min_value=1.0, value=9.0, step=0.5, format="%.1f")
mu_soil_friction_forerunner = st.sidebar.number_input("Forerunner-Soil Friction Coeff (μ_soil)", min_value=0.0, value=0.25, step=0.01, format="%.2f")
theta_mudline_deg_initial = st.sidebar.number_input("Initial Mooring Angle at Mudline (θ_m, degrees, for taut line approx.)", min_value=0.0, value=5.0, step=1.0, format="%.1f")

# Caisson Sizing
st.sidebar.subheader("Suction Caisson Sizing")
L_D_ratio_caisson = st.sidebar.number_input("Target Caisson Length-to-Diameter Ratio (L/D)", min_value=1.0, value=3.2, step=0.1, format="%.1f")
padeye_loc_ratio_z_a_L = st.sidebar.number_input("Padeye Location from Top (z_a/L)", min_value=0.1, value=0.60, step=0.05, format="%.2f") # AB2018: 0.5 for clay, 2/3 for sand
t_wall_D_ratio_caisson = st.sidebar.number_input("Caisson Wall Thickness to Diameter Ratio (D/t_wall)", min_value=10.0, value=70.0, step=5.0, format="%.1f")
rho_steel_submerged_kg_m3 = st.sidebar.number_input("Submerged Steel Density (kg/m³)", value=6825.0, step=10.0, format="%.1f", help="Approx (7850 - 1025)")


if soil_type == "Clay":
    s_u_profile_clay = st.sidebar.selectbox("Clay Undrained Shear Strength (s_u) Profile", ["Constant", "Linearly Increasing"], index=0)
    if s_u_profile_clay == "Constant":
        s_u_avg_clay_kPa_input = st.sidebar.number_input("Average s_u (kPa)", min_value=1.0, value=30.0, step=1.0, format="%.1f")
        s_u_tip_clay_kPa_input = s_u_avg_clay_kPa_input # For constant profile
        s_u0_clay_kPa_input = s_u_avg_clay_kPa_input
        k_su_increase_clay_kPa_m_input = 0.0
    else: # Linearly Increasing
        s_u0_clay_kPa_input = st.sidebar.number_input("s_u at Mudline (s_u0, kPa)", min_value=1.0, value=15.0, step=1.0, format="%.1f")
        k_su_increase_clay_kPa_m_input = st.sidebar.number_input("s_u Increase Rate (kPa/m)", min_value=0.0, value=2.0, step=0.1, format="%.1f")
        # s_u_avg and s_u_tip will be calculated based on caisson length L later
    
    N_p_lateral_clay = st.sidebar.number_input("Lateral Bearing Factor Clay (N_p)", min_value=1.0, value=10.0, step=0.5, format="%.1f") 
    Nc_tip_clay = st.sidebar.number_input("Tip Bearing Factor Clay (N_c,tip)", min_value=1.0, value=9.0, step=0.5, format="%.1f")
    alpha_e_clay = st.sidebar.number_input("External Shaft Friction Coeff. Clay (α_e)", min_value=0.1, value=0.7, step=0.05, format="%.2f")
    alpha_i_clay = st.sidebar.number_input("Internal Shaft Friction Coeff. Clay (α_i)", min_value=0.1, value=0.5, step=0.05, format="%.2f")
    gamma_prime_soil_plug_clay_kNm3 = st.sidebar.number_input("Submerged Unit Weight of Soil Plug (γ'_plug, kN/m³)", min_value=1.0, value=8.0, step=0.5, format="%.1f")


elif soil_type == "Sand":
    phi_sand_deg = st.sidebar.number_input("Sand Internal Friction Angle (φ', degrees)", min_value=10.0, value=30.0, step=1.0, format="%.1f")
    gamma_prime_sand_kNm3 = st.sidebar.number_input("Sand Submerged Unit Weight (γ', kN/m³)", min_value=5.0, value=9.0, step=0.5, format="%.1f")
    K_tan_delta_ext_sand = st.sidebar.number_input("(K tan δ)_e for Sand", min_value=0.1, value=0.5, step=0.05, format="%.2f") 
    K_tan_delta_int_sand = st.sidebar.number_input("(K tan δ)_i for Sand", min_value=0.1, value=0.3, step=0.05, format="%.2f") 

# --- Calculation Trigger ---
if st.sidebar.button("Run Analysis", use_container_width=True):
    results = {}
    st.header("Results")

    # --- 1. Wind Load Calculations ---
    with st.expander("1. Wind Load Calculations", expanded=True):
        U10_50yr, U10_1yr = calculate_extreme_wind_speeds(weibull_lambda, weibull_k_shape)
        results['U10_50yr'] = U10_50yr
        results['U10_1yr'] = U10_1yr
        st.write(f"50-year Extreme Wind Speed (U_10,50yr): {U10_50yr:.2f} m/s")
        st.write(f"1-year Extreme Wind Speed (U_10,1yr): {U10_1yr:.2f} m/s")

        sigma_Uc_val = calculate_sigma_Uc(U10_1yr)
        results['sigma_Uc'] = sigma_Uc_val
        st.write(f"Characteristic Std. Dev. of Wind Speed (σ_U,c): {sigma_Uc_val:.2f} m/s")

        u_EOG_val = calculate_eog_speed(U10_1yr, U_R_turbine, sigma_Uc_val, Lk_turbulence, D_rotor)
        results['u_EOG'] = u_EOG_val
        st.write(f"Extreme Operating Gust Speed (u_EOG) at U_R: {u_EOG_val:.2f} m/s")

        C_T_EOG = get_thrust_coefficient(U_R_turbine + u_EOG_val, U_R_turbine, C_T_method_choice, U_cut_in_turbine, U_cut_out_turbine)
        F_u_EOG = calculate_rotor_thrust_force(rho_air, D_rotor, C_T_EOG, U_R_turbine + u_EOG_val) / 1e6 # MN
        results['F_u_EOG_MN'] = F_u_EOG
        st.write(f"Rotor Thrust (EOG at U_R, F_u,EOG): {F_u_EOG:.2f} MN (C_T={C_T_EOG:.2f} for U_eff={(U_R_turbine + u_EOG_val):.2f} m/s)")
        
        F_DT_EOG = calculate_tower_drag_force(rho_air, C_DT_tower, U_R_turbine + u_EOG_val, z_hub, D_tower_bottom, D_tower_top, gamma_shear) / 1e6 # MN
        results['F_DT_EOG_MN'] = F_DT_EOG
        st.write(f"Tower Drag (at U_R + u_EOG, F_DT,EOG): {F_DT_EOG:.2f} MN")

        F_DT_U50_N = calculate_tower_drag_force(rho_air, C_DT_tower, U10_50yr, z_hub, D_tower_bottom, D_tower_top, gamma_shear)
        results['F_DT_U50_MN'] = F_DT_U50_N / 1e6
        st.write(f"Tower Drag (at U_10,50yr, F_DT,U50): {results['F_DT_U50_MN']:.2f} MN")
        
        F_u_U50_total = calculate_shutdown_thrust_drag(rho_air, U10_50yr, D_rotor, C_D_rotor_parked, F_DT_U50_N) / 1e6 # MN
        results['F_u_U50_total_MN'] = F_u_U50_total
        st.write(f"Total Shutdown Wind Load (Rotor Drag + Tower Drag at U_10,50yr, F_u,U50_total): {F_u_U50_total:.2f} MN")


    # --- 2. Wave Load Calculations ---
    with st.expander("2. Wave Load Calculations", expanded=True):
        H_S1, T_S1, H_m1, T_m1, H_S50, T_S50, H_m50, T_m50 = get_wave_design_parameters(H_S50_known)
        results.update({'H_S1':H_S1, 'T_S1':T_S1, 'H_m1':H_m1, 'T_m1':T_m1, 'H_S50':H_S50, 'T_S50':T_S50, 'H_m50':H_m50, 'T_m50':T_m50})
        st.write(f"1-yr Waves: H_S1={H_S1:.2f}m, T_S1={T_S1:.2f}s; H_m1={H_m1:.2f}m, T_m1={T_m1:.2f}s")
        st.write(f"50-yr Waves: H_S50={H_S50:.2f}m, T_S50={T_S50:.2f}s; H_m50={H_m50:.2f}m, T_m50={T_m50:.2f}s")

        omega_1yr = 2.0 * PI / T_m1 if T_m1 > 0 else 0.0
        k_1yr = solve_dispersion_relation_iterative(omega_1yr, S_water_depth)
        F_D_max_1yr_N, F_I_max_1yr_N = calculate_morison_wave_forces(rho_water, D_spar, C_D_spar_wave, C_m_spar_wave, H_m1, T_m1, S_water_depth, B_spar_draft, k_1yr)
        F_w_1yr_MN = (F_D_max_1yr_N + F_I_max_1yr_N) / 1e6 # MN
        results.update({'k_1yr':k_1yr, 'F_D_max_1yr_MN':F_D_max_1yr_N/1e6, 'F_I_max_1yr_MN':F_I_max_1yr_N/1e6, 'F_w_1yr_MN':F_w_1yr_MN})
        st.write(f"1-yr EWH Load (F_w,1yr): {F_w_1yr_MN:.2f} MN (Drag: {F_D_max_1yr_N/1e6:.2f} MN, Inertia: {F_I_max_1yr_N/1e6:.2f} MN, k={k_1yr:.4f} rad/m)")

        omega_50yr = 2.0 * PI / T_m50 if T_m50 > 0 else 0.0
        k_50yr = solve_dispersion_relation_iterative(omega_50yr, S_water_depth)
        F_D_max_50yr_N, F_I_max_50yr_N = calculate_morison_wave_forces(rho_water, D_spar, C_D_spar_wave, C_m_spar_wave, H_m50, T_m50, S_water_depth, B_spar_draft, k_50yr)
        F_w_50yr_MN = (F_D_max_50yr_N + F_I_max_50yr_N) / 1e6 # MN
        results.update({'k_50yr':k_50yr, 'F_D_max_50yr_MN':F_D_max_50yr_N/1e6, 'F_I_max_50yr_MN':F_I_max_50yr_N/1e6, 'F_w_50yr_MN':F_w_50yr_MN})
        st.write(f"50-yr EWH Load (F_w,50yr): {F_w_50yr_MN:.2f} MN (Drag: {F_D_max_50yr_N/1e6:.2f} MN, Inertia: {F_I_max_50yr_N/1e6:.2f} MN, k={k_50yr:.4f} rad/m)")

    # --- 3. Current Load Calculation ---
    with st.expander("3. Current Load Calculation", expanded=True):
        F_C_N = calculate_current_force(rho_water, D_spar, C_D_spar_current, v_current_speed, B_spar_draft)
        results['F_C_MN'] = F_C_N / 1e6
        st.write(f"Total Current Load (F_C): {results['F_C_MN']:.2f} MN")

    # --- 4. ULS Load Combinations (at floater padeye) ---
    with st.expander("4. ULS Load Combinations (at floater padeye)", expanded=True):
        F_E1_MN = results['F_u_U50_total_MN'] + results['F_w_50yr_MN'] + results['F_C_MN']
        results['F_E1_MN'] = F_E1_MN
        st.write(f"Load Case E-1 (50yr wind_shut + 50yr wave + current): {F_E1_MN:.2f} MN")

        F_wind_EOG_total_MN = results['F_u_EOG_MN'] + results['F_DT_EOG_MN']
        F_E2_MN = F_wind_EOG_total_MN + results['F_w_1yr_MN'] + results['F_C_MN']
        results['F_E2_MN'] = F_E2_MN
        st.write(f"Load Case E-2 (EOG@U_R wind + 1yr wave + current): {F_E2_MN:.2f} MN")
        
        F_ULS_floater_MN = max(F_E1_MN, F_E2_MN)
        results['F_ULS_floater_MN'] = F_ULS_floater_MN
        st.markdown(f"**Dominant ULS Horizontal Load at Floater Padeye: {F_ULS_floater_MN:.2f} MN**")
        T_mudline_N = F_ULS_floater_MN * 1e6 

    # --- 5. Load Transfer to Anchor ---
    with st.expander("5. Load Transfer to Anchor", expanded=True):
        st.write("Anchor padeye load (T_a, θ_a) will be determined iteratively during caisson sizing.")
        st.write(f"Using ULS Floater Load as initial mudline tension T_m = {T_mudline_N/1e3:.0f} kN at θ_m = {theta_mudline_deg_initial:.1f}°")


    # --- 6. Suction Caisson Sizing (Iterative) ---
    with st.expander("6. Suction Caisson Sizing", expanded=True):
        st.write(f"Target L/D = {L_D_ratio_caisson:.2f}, Padeye z_a/L = {padeye_loc_ratio_z_a_L:.2f}")
        
        D_caisson_current_m = 2.0 
        D_increment_m = 0.05 # Make increment smaller for finer search
        max_D_caisson_m = 20.0 
        found_D = False
        
        final_FP_val = float('inf') # Initialize here for access in error message if not found
        
        iterations_data = []

        for i in range(int((max_D_caisson_m - D_caisson_current_m)/D_increment_m) + 1):
            L_caisson_current_m = D_caisson_current_m * L_D_ratio_caisson
            z_a_padeye_current_m = padeye_loc_ratio_z_a_L * L_caisson_current_m

            current_s_u_avg_kPa = 0.0
            current_s_u_tip_kPa = 0.0
            gamma_prime_soil_plug_Nm3 = 0.0

            if soil_type == "Clay":
                if s_u_profile_clay == "Constant":
                    current_s_u_avg_kPa = s_u_avg_clay_kPa_input
                    current_s_u_tip_kPa = s_u_tip_clay_kPa_input
                else: 
                    current_s_u_avg_kPa = s_u0_clay_kPa_input + k_su_increase_clay_kPa_m_input * (L_caisson_current_m / 2.0)
                    current_s_u_tip_kPa = s_u0_clay_kPa_input + k_su_increase_clay_kPa_m_input * L_caisson_current_m
                gamma_prime_soil_plug_Nm3 = gamma_prime_soil_plug_clay_kNm3 * 1000.0
                s_u_for_Qavg_Pa = current_s_u_avg_kPa * 1000.0 # Use avg s_u for Q_avg calculation consistency
                gamma_prime_for_Qavg_Nm3 = 0.0
            
            elif soil_type == "Sand":
                gamma_prime_soil_plug_Nm3 = gamma_prime_sand_kNm3 * 1000.0
                gamma_prime_for_Qavg_Nm3 = gamma_prime_sand_kNm3 * 1000.0
                s_u_for_Qavg_Pa = 0.0
            
            Q_avg_val_Nm = calculate_Q_avg_forerunner(soil_type, s_u_for_Qavg_Pa, gamma_prime_for_Qavg_Nm3, z_a_padeye_current_m, A_b_forerunner, Nc_forerunner_bearing)
            
            T_a_N, theta_a_deg = solve_anchor_padeye_load_system(T_mudline_N, theta_mudline_deg_initial, z_a_padeye_current_m, Q_avg_val_Nm, mu_soil_friction_forerunner)
            
            H_u_anchor_N = T_a_N * math.cos(to_radians(theta_a_deg))
            V_u_anchor_N = T_a_N * math.sin(to_radians(theta_a_deg))
            if V_u_anchor_N < 0: V_u_anchor_N = 0.0

            W_prime_caisson_N, W_prime_plug_N, t_wall_m, D_i_m = estimate_caisson_weights(
                D_caisson_current_m, L_caisson_current_m, t_wall_D_ratio_caisson, 
                rho_steel_submerged_kg_m3, gamma_prime_soil_plug_Nm3, soil_type
            )

            H_m_cap_N, V_m_cap_N = 0.0, 0.0
            debug_cap_info = {}
            if soil_type == "Clay":
                H_m_cap_N, V_m_cap_N, F_ext_f, F_int_f, F_rev_eb, Vm1, Vm2, Vm3 = calculate_caisson_capacity_clay(
                    L_caisson_current_m, D_caisson_current_m, D_i_m, current_s_u_avg_kPa, current_s_u_tip_kPa,
                    N_p_lateral_clay, Nc_tip_clay, alpha_e_clay, alpha_i_clay,
                    W_prime_caisson_N, W_prime_plug_N
                )
                debug_cap_info = {'Vm1_kN':Vm1/1e3, 'Vm2_kN':Vm2/1e3, 'Vm3_kN':Vm3/1e3, 'F_ext_f_kN':F_ext_f/1e3, 'F_int_f_kN':F_int_f/1e3, 'F_rev_eb_kN':F_rev_eb/1e3}
            elif soil_type == "Sand":
                N_q_s = calculate_Nq_sand(phi_sand_deg)
                H_m_cap_N, V_m_cap_N = calculate_caisson_capacity_sand(
                    L_caisson_current_m, D_caisson_current_m, D_i_m, gamma_prime_sand_kNm3 * 1000.0, N_q_s, 
                    W_prime_caisson_N, K_tan_delta_ext_sand, K_tan_delta_int_sand
                )
                debug_cap_info = {'Nq_sand': N_q_s}

            final_FP_val, a_exp, b_exp = check_caisson_failure_criterion(H_u_anchor_N, V_u_anchor_N, H_m_cap_N, V_m_cap_N, L_caisson_current_m, D_caisson_current_m)
            
            iterations_data.append({
                "D (m)": D_caisson_current_m, "L (m)": L_caisson_current_m, "z_a (m)": z_a_padeye_current_m,
                "T_a (kN)": T_a_N/1e3, "θ_a (deg)": theta_a_deg,
                "H_u (kN)": H_u_anchor_N/1e3, "V_u (kN)": V_u_anchor_N/1e3,
                "H_m_cap (kN)": H_m_cap_N/1e3, "V_m_cap (kN)": V_m_cap_N/1e3,
                "FP": final_FP_val, "a": a_exp, "b": b_exp,
                "W'_c (kN)": W_prime_caisson_N/1e3, "W'_plug (kN)": W_prime_plug_N/1e3,
                **debug_cap_info
            })

            if final_FP_val <= 1.005: # Allow a small tolerance for FP
                results['D_caisson_final_m'] = D_caisson_current_m
                results['L_caisson_final_m'] = L_caisson_current_m
                results['t_wall_final_m'] = t_wall_m
                results['z_a_padeye_final_m'] = z_a_padeye_current_m
                results['T_a_final_kN'] = T_a_N / 1e3
                results['theta_a_final_deg'] = theta_a_deg
                results['H_u_anchor_final_kN'] = H_u_anchor_N / 1e3
                results['V_u_anchor_final_kN'] = V_u_anchor_N / 1e3
                results['H_m_cap_final_kN'] = H_m_cap_N / 1e3
                results['V_m_cap_final_kN'] = V_m_cap_N / 1e3
                results['FP_final'] = final_FP_val
                results['a_exp_final'] = a_exp
                results['b_exp_final'] = b_exp
                results['W_prime_caisson_final_kN'] = W_prime_caisson_N / 1e3
                results['W_prime_plug_final_kN'] = W_prime_plug_N / 1e3
                if soil_type == "Clay": results.update(debug_cap_info)
                if soil_type == "Sand": results.update(debug_cap_info)
                found_D = True
                break 
            
            D_caisson_current_m += D_increment_m

        if found_D:
            st.success(f"Found suitable caisson dimensions (FP={results['FP_final']:.3f} <= 1.0):")
            st.markdown(f"""
            - **Min. Caisson Diameter (D): {results['D_caisson_final_m']:.2f} m**
            - **Caisson Length (L): {results['L_caisson_final_m']:.2f} m** (L/D = {results['L_caisson_final_m']/results['D_caisson_final_m']:.2f})
            - Wall Thickness (t_w): {results['t_wall_final_m']*1000:.1f} mm
            - Padeye Depth (z_a): {results['z_a_padeye_final_m']:.2f} m
            - Anchor Padeye Tension (T_a): {results['T_a_final_kN']:.1f} kN at {results['theta_a_final_deg']:.1f}°
            - Design Loads on Anchor: H_u = {results['H_u_anchor_final_kN']:.1f} kN, V_u = {results['V_u_anchor_final_kN']:.1f} kN
            - Capacities: H_m = {results['H_m_cap_final_kN']:.1f} kN, V_m = {results['V_m_cap_final_kN']:.1f} kN
            - Failure Criterion (FP): {results['FP_final']:.3f} (Exponents: a={results['a_exp_final']:.2f}, b={results['b_exp_final']:.2f})
            - Submerged Weights: Caisson W'_c = {results['W_prime_caisson_final_kN']:.1f} kN, Soil Plug W'_plug = {results['W_prime_plug_final_kN']:.1f} kN
            """)
            if soil_type == "Clay" and 'Vm1_kN' in results:
                 st.write(f"Clay Vm details (kN): Vm1={results.get('Vm1_kN',0):.1f}, Vm2={results.get('Vm2_kN',0):.1f}, Vm3={results.get('Vm3_kN',0):.1f} (Min chosen)")
            if soil_type == "Sand" and 'Nq_sand' in results:
                 st.write(f"Sand Nq factor: {results.get('Nq_sand',0):.2f}")

        else:
            st.error(f"Could not find suitable caisson diameter up to {max_D_caisson_m:.1f} m for L/D={L_D_ratio_caisson:.1f}. Last FP = {final_FP_val:.3f}")

        if iterations_data:
            st.subheader("Sizing Iteration Details (Last 10 or all if fewer)")
            display_df = iterations_data[-10:] # Show last 10 iterations or all if fewer
            # Select key columns for display to avoid overly wide table
            cols_to_show = ["D (m)", "L (m)", "H_u (kN)", "V_u (kN)", "H_m_cap (kN)", "V_m_cap (kN)", "FP"]
            if soil_type == "Clay":
                cols_to_show.extend(['Vm1_kN', 'Vm2_kN', 'Vm3_kN'])
            elif soil_type == "Sand":
                cols_to_show.append('Nq_sand')

            # Filter dataframe to only include existing columns from cols_to_show
            filtered_df_data = []
            for row in display_df:
                filtered_row = {col: row.get(col, 'N/A') for col in cols_to_show}
                filtered_df_data.append(filtered_row)
            
            if filtered_df_data:
                st.dataframe(filtered_df_data, height=300)


    # --- Disclaimer ---
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Disclaimer:** This tool is for educational and preliminary estimation purposes only. 
    Results should be verified by qualified professionals using detailed engineering analysis and relevant standards.
    """)

else:
    st.info("Adjust parameters in the sidebar and click 'Run Analysis'.")
