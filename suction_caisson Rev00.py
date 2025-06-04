import streamlit as st
import math
import numpy as np
from scipy.optimize import fsolve

# --- Constants ---
G_ACCEL = 9.81
RHO_AIR_DEFAULT = 1.225  # kg/m^3
RHO_WATER_DEFAULT = 1025 # kg/m^3 (can be adjusted for seawater)
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

def get_thrust_coefficient(U_wind, U_R_turbine, C_T_method="frohboese_ab2018", U_cut_in=4, U_cut_out=25):
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
            return C_T_rated * (U_R_turbine / U_wind)**2 # Simplified, assumes constant power output by varying pitch
            
    elif C_T_method == "simplified_mn2022":
        if U_wind <= U_R_turbine:
            return 7 / U_wind if U_wind > 0 else 0 # MN2022 Eq 16 (u_bar = U_wind)
        else:
            return (7 * U_R_turbine**2) / U_wind**3 if U_wind > 0 else 0 # MN2022 Eq 18
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
    
    T_S50 = 11.1 * math.sqrt(H_S50_known / g) if H_S50_known > 0 and g > 0 else 0
    T_S1 = 11.1 * math.sqrt(H_S1 / g) if H_S1 > 0 and g > 0 else 0
    
    N_waves_50 = (3 * 3600) / T_S50 if T_S50 > 0 else 0
    N_waves_1 = (3 * 3600) / T_S1 if T_S1 > 0 else 0
    
    H_m50 = H_S50_known * math.sqrt(0.5 * math.log(N_waves_50)) if N_waves_50 > 0 and H_S50_known > 0 else 0
    H_m1 = H_S1 * math.sqrt(0.5 * math.log(N_waves_1)) if N_waves_1 > 0 and H_S1 > 0 else 0
    
    T_m50 = 11.1 * math.sqrt(H_m50 / g) if H_m50 > 0 and g > 0 else 0
    T_m1 = 11.1 * math.sqrt(H_m1 / g) if H_m1 > 0 and g > 0 else 0
    
    return H_S1, T_S1, H_m1, T_m1, H_S50_known, T_S50, H_m50, T_m50

def dispersion_equation(k, omega, S_water_depth, g=G_ACCEL):
    if k < 0: k = 1e-6 # k must be positive
    return omega**2 - g * k * math.tanh(k * S_water_depth)

def solve_dispersion_relation_iterative(omega_wave, S_water_depth, g=G_ACCEL):
    if omega_wave == 0: return 0
    # Initial guess for k (deep water approximation: k_approx = omega^2 / g)
    k_initial_guess = omega_wave**2 / g
    try:
        k_solution, = fsolve(dispersion_equation, k_initial_guess, args=(omega_wave, S_water_depth, g), xtol=1e-8)
        return k_solution if k_solution > 0 else 1e-6 # Ensure positive k
    except Exception:
        # Fallback for shallow water: k_approx = omega / sqrt(g*S)
        if S_water_depth > 0 and g > 0:
            k_fallback = omega_wave / math.sqrt(g * S_water_depth)
            return k_fallback
        return 1e-6


def calculate_morison_wave_forces(rho_water, D_spar, C_D_spar, C_m_spar, H_m_wave, T_m_wave, S_water_depth, B_spar_draft, k_wave_number):
    # AB2018 Eq 20, 21 (integrated forms P_D, P_I)
    if T_m_wave == 0 or k_wave_number == 0 or math.sinh(k_wave_number * S_water_depth) == 0:
        return 0, 0

    # Max Drag Force (eta = H_m/2 at t=0 for u(z,0))
    # P_D term from AB2018 Eq 20
    # Integration from -B to H_m/2
    # cosh(k(S+z))^2 dz
    # P_D = [sinh(2k(S+z))/(4k) + z/2] from -B to H_m/2
    # Note: AB2018 uses slightly different P_D formulation involving exponentials. Let's use that.
    # P_D = (1/(8k)) * (exp(2k(S+Hm/2)) - exp(-2k(S+Hm/2)) - exp(2k(S-B)) + exp(-2k(S-B))) + Hm/4 + B/2
    
    # Check for large kS values to prevent overflow in sinh/cosh
    kS = k_wave_number * S_water_depth
    if kS > 700 : # sinh(700) is already huge
        sinh_kS_sq = float('inf')
    else:
        sinh_kS_sq = math.sinh(k_wave_number * S_water_depth)**2
    
    if sinh_kS_sq == 0: return 0,0

    # P_D from AB2018 Eq 20
    term_exp1 = math.exp(2 * k_wave_number * (S_water_depth + H_m_wave / 2))
    term_exp2 = math.exp(-2 * k_wave_number * (S_water_depth + H_m_wave / 2)) # This might be large if S+Hm/2 is negative, but z is positive upwards
    term_exp3 = math.exp(2 * k_wave_number * (S_water_depth - B_spar_draft))
    term_exp4 = math.exp(-2 * k_wave_number * (S_water_depth - B_spar_draft))
    
    P_D = (1 / (8 * k_wave_number)) * (term_exp1 - term_exp2 - term_exp3 + term_exp4) + H_m_wave / 4 + B_spar_draft / 2

    F_D_max = 0.5 * rho_water * D_spar * C_D_spar * (PI**2 * H_m_wave**2) / (T_m_wave**2 * sinh_kS_sq) * P_D

    # Max Inertia Force (eta = 0 at t=Tm/4 for u_dot(z, Tm/4))
    # P_I term from AB2018 Eq 21
    # Integration from -B to 0
    # cosh(k(S+z)) dz = [sinh(k(S+z))/k] from -B to 0
    # P_I = (sinh(kS) - sinh(k(S-B)))/k
    P_I = (math.sinh(k_wave_number * S_water_depth) - math.sinh(k_wave_number * (S_water_depth - B_spar_draft))) / k_wave_number
    A_p_spar = PI * (D_spar/2)**2
    F_I_max = C_m_spar * rho_water * A_p_spar * (2 * PI**2 * H_m_wave) / (T_m_wave**2 * math.sinh(k_wave_number*S_water_depth)) * P_I
    
    return F_D_max, F_I_max

# --- Current Load Calculation (Ref: AB2018 Sec 2.1.4; MN2022 Eq 41) ---
def calculate_current_force(rho_water, D_spar, C_D_current_spar, v_current_speed, B_spar_draft):
    # AB2018 Eq 25
    return 0.5 * rho_water * D_spar * C_D_current_spar * v_current_speed**2 * B_spar_draft

# --- Load Transfer to Anchor (Ref: AB2018 Sec 3.4.3; MN2022 Sec 5.4) ---
def calculate_Q_avg_forerunner(soil_type, s_u_avg_clay, gamma_prime_sand, z_a_padeye, A_b_forerunner, Nc_forerunner_bearing):
    # AB2018 Eq 52 (clay), 53 (sand)
    if soil_type == "Clay":
        # For clay, Q_av = Nc * s_u(z) * A_b / z_a. Integral of s_u(z) from 0 to z_a is s_u_avg * z_a
        # So, z_a * Q_av = A_b * Nc * s_u_avg * z_a. Thus Q_av = A_b * Nc * s_u_avg
        # However, the paper's Eq 52 is z_a * Q_av = A_b * Nc * integral(s_u(z)dz)
        # If s_u is constant s_u_avg, then integral(s_u(z)dz) = s_u_avg * z_a
        # So Q_av = A_b * Nc_forerunner_bearing * s_u_avg_clay
        # This interpretation seems more consistent with the formula structure.
        # Let's assume s_u(z) is constant s_u_avg over the forerunner depth for simplicity.
        # Then integral(s_u(z)dz) from 0 to z_a is s_u_avg_clay * z_a
        # z_a * Q_av = A_b_forerunner * Nc_forerunner_bearing * (s_u_avg_clay * z_a)
        # This implies Q_av = A_b_forerunner * Nc_forerunner_bearing * s_u_avg_clay
        # Let's re-evaluate based on the equations T_a/2 * (theta_a^2 - theta_m^2) = z_a * Q_av
        # Q_av is average soil resistance PER UNIT LENGTH of forerunner.
        # So, for clay: Q_av = A_b_forerunner * Nc_forerunner_bearing * s_u_avg_clay (Force/Length)
        return A_b_forerunner * Nc_forerunner_bearing * s_u_avg_clay
    elif soil_type == "Sand":
        # For sand: z_a * Q_av = A_b_forerunner * Nc_forerunner_bearing * integral(gamma_prime*z dz) from 0 to z_a
        # integral = gamma_prime * z_a^2 / 2
        # So Q_av = A_b_forerunner * Nc_forerunner_bearing * gamma_prime_sand * z_a / 2
        return A_b_forerunner * Nc_forerunner_bearing * gamma_prime_sand * z_a / 2
    return 0

def anchor_load_equations(vars_in, T_mudline, theta_mudline_rad, z_a_padeye, Q_avg, mu_soil_friction):
    T_a, theta_a_rad = vars_in
    if T_a <=0: T_a = 1e-3 # Avoid division by zero or log of non-positive

    # AB2018 Eq 50, 51 (modified for direct solving)
    # Eq 50: T_a/2 * (theta_a^2 - theta_m^2) - z_a * Q_av = 0
    # Eq 51: T_m / T_a - exp(mu * (theta_a - theta_m)) = 0  => T_m - T_a * exp(mu * (theta_a - theta_m)) = 0
    eq1 = (T_a / 2) * (theta_a_rad**2 - theta_mudline_rad**2) - z_a_padeye * Q_avg
    eq2 = T_mudline - T_a * math.exp(mu_soil_friction * (theta_a_rad - theta_mudline_rad))
    return (eq1, eq2)

def solve_anchor_padeye_load_system(T_mudline, theta_mudline_deg, z_a_padeye, Q_avg, mu_soil_friction):
    theta_mudline_rad = to_radians(theta_mudline_deg)
    # Initial guesses: T_a slightly less than T_m, theta_a slightly more than theta_m
    initial_guesses = [T_mudline * 0.95, theta_mudline_rad + to_radians(1)]
    if Q_avg <=0 or z_a_padeye <=0 : # No soil resistance, T_a = T_m, theta_a = theta_m
        return T_mudline, theta_mudline_deg

    try:
        solution = fsolve(anchor_load_equations, initial_guesses, args=(T_mudline, theta_mudline_rad, z_a_padeye, Q_avg, mu_soil_friction), xtol=1e-6)
        T_a_sol, theta_a_rad_sol = solution
        return T_a_sol, to_degrees(theta_a_rad_sol)
    except Exception as e:
        # st.warning(f"Solver for anchor padeye load failed: {e}. Returning mudline values.")
        return T_mudline, theta_mudline_deg


# --- Suction Caisson Sizing (Ref: AB2018 Sec 3.4) ---
def estimate_caisson_weights(D_caisson, L_caisson, t_wall_ratio, rho_steel_submerged_factor=68.5e3/G_ACCEL, gamma_prime_soil_plug=9e3, soil_type="Clay"):
    # rho_steel_submerged_factor: submerged density of steel (approx 7850 - 1025 = 6825 kg/m3 => 68.5 kN/m3 / g)
    t_wall = D_caisson / t_wall_ratio if t_wall_ratio > 0 else 0.05 # Default 5cm if ratio is bad
    D_i_caisson = D_caisson - 2 * t_wall
    
    Volume_steel = PI * ( (D_caisson/2)**2 - (D_i_caisson/2)**2 ) * L_caisson
    W_prime_caisson = Volume_steel * rho_steel_submerged_factor * G_ACCEL # Submerged weight in N
    
    W_prime_plug = 0
    if soil_type == "Clay" or soil_type == "Sand": # Soil plug relevant for both
        Area_plug = PI * (D_i_caisson/2)**2
        W_prime_plug = Area_plug * L_caisson * gamma_prime_soil_plug # gamma_prime_soil_plug in N/m3
        
    return W_prime_caisson, W_prime_plug, t_wall, D_i_caisson

def calculate_Nq_sand(phi_sand_degrees):
    # AB2018 Eq 48
    phi_rad = to_radians(phi_sand_degrees)
    return math.exp(PI * math.tan(phi_rad)) * math.tan(PI/4 + phi_rad/2)**2

def calculate_caisson_capacity_clay(L_caisson, D_e_caisson, D_i_caisson, avg_s_u, tip_s_u, N_p_lateral_clay, Nc_tip_clay, alpha_e_clay, alpha_i_clay, W_prime_caisson, W_prime_plug):
    # AB2018 Sec 3.4.1
    # Horizontal Capacity Hm (Eq 42)
    H_m = L_caisson * D_e_caisson * N_p_lateral_clay * avg_s_u * 1000 # avg_s_u in kPa to Pa
    
    # Vertical Capacities Vm1, Vm2, Vm3
    A_se_external_shaft_area = PI * D_e_caisson * L_caisson
    A_si_internal_shaft_area = PI * D_i_caisson * L_caisson
    A_e_external_cross_section_area = PI * (D_e_caisson/2)**2
    
    F_ext_friction = A_se_external_shaft_area * alpha_e_clay * avg_s_u * 1000
    F_int_friction = A_si_internal_shaft_area * alpha_i_clay * avg_s_u * 1000 # Assuming avg_s_u for internal too
    F_rev_end_bearing = Nc_tip_clay * tip_s_u * 1000 * A_e_external_cross_section_area # tip_s_u at caisson tip
    
    V_m1 = W_prime_caisson + F_ext_friction + F_rev_end_bearing # Passive suction + rev end bearing
    V_m2 = W_prime_caisson + F_ext_friction + F_int_friction    # No passive suction, caisson pullout (internal friction)
    V_m3 = W_prime_caisson + F_ext_friction + W_prime_plug      # No passive suction, plug pullout
    
    V_m = min(V_m1, V_m2, V_m3)
    return H_m, V_m, F_ext_friction, F_int_friction, F_rev_end_bearing, V_m1, V_m2, V_m3

def y_x_sand_capacity(x_val):
    # AB2018 Eq 49 related term; MN2022 Eq 45 related term y(x) = exp(-x) - 1 + x
    return math.exp(-x_val) - 1 + x_val

def calculate_caisson_capacity_sand(L_caisson, D_e_caisson, D_i_caisson, gamma_prime_sand, N_q_sand, W_prime_caisson, K_tan_delta_ext, K_tan_delta_int):
    # AB2018 Sec 3.4.2
    # Horizontal Capacity Hm (Eq 46) - Note: Q_av from Miedema. Simpler form: 0.5 * D_e * N_q * gamma_prime * L^2 (approx)
    # Using AB2018 notation: H_m = D_e * N_q * (0.5 * gamma_prime * L^2) - this seems to be for a strip.
    # The formula in AB2018 is H_m,sand = 0.5 * A_b * N_q * gamma_prime * L^2. A_b is not caisson area.
    # Let's use a common formulation for lateral capacity of piles in sand (e.g., Broms or API)
    # For simplicity, using a Rankine passive earth pressure approach or similar.
    # AB2018 Eq 46: H_m,sand = L * Q_av where Q_av = D_e * N_q * integral(gamma_prime*z dz) / L = 0.5 * D_e * N_q * gamma_prime * L
    # So H_m,sand = 0.5 * D_e_caisson * N_q_sand * gamma_prime_sand * L_caisson**2
    H_m = 0.5 * D_e_caisson * N_q_sand * gamma_prime_sand * L_caisson**2
    
    # Vertical Capacity Vm (Eq 49)
    # Z_e = D_e / (4 * K_tan_delta_ext), Z_i = D_i / (4 * K_tan_delta_int)
    # h/Z_e = L / Z_e, h/Z_i = L / Z_i
    if K_tan_delta_ext == 0 or K_tan_delta_int == 0 : return H_m, 0 # Avoid division by zero
    
    Z_e = D_e_caisson / (4 * K_tan_delta_ext)
    Z_i = D_i_caisson / (4 * K_tan_delta_int)
    
    term_ext_friction = 0
    if Z_e > 0:
        h_div_Z_e = L_caisson / Z_e
        term_ext_friction = gamma_prime_sand * Z_e**2 * y_x_sand_capacity(h_div_Z_e) * K_tan_delta_ext * PI * D_e_caisson
        
    term_int_friction = 0
    if Z_i > 0:
        h_div_Z_i = L_caisson / Z_i
        term_int_friction = gamma_prime_sand * Z_i**2 * y_x_sand_capacity(h_div_Z_i) * K_tan_delta_int * PI * D_i_caisson
        
    V_m = W_prime_caisson + term_ext_friction + term_int_friction
    return H_m, V_m


def check_caisson_failure_criterion(H_u_on_anchor, V_u_on_anchor, H_m_capacity, V_m_capacity, L_caisson, D_caisson):
    # AB2018 Eq 40, 41
    if D_caisson == 0 or V_m_capacity == 0 or H_m_capacity == 0: return float('inf'), 0, 0
    
    a_exp = (L_caisson / D_caisson) + 0.5
    b_exp = (L_caisson / (3 * D_caisson)) + 4.5
    
    # Ensure non-negative bases for exponentiation
    term_H = H_u_on_anchor / H_m_capacity
    term_V = V_u_on_anchor / V_m_capacity

    if term_H < 0: term_H = 0
    if term_V < 0: term_V = 0
    
    FP = (term_H)**a_exp + (term_V)**b_exp
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
gamma_shear = st.sidebar.number_input("Wind Shear Exponent (γ)", min_value=0.05, value=1/7, step=0.01, format="%.3f")
# Wave
H_S50_known = st.sidebar.number_input("50-year Significant Wave Height (H_S50, m)", min_value=1.0, value=10.0, step=0.5, format="%.1f")
rho_water = st.sidebar.number_input("Sea Water Density (ρ_water, kg/m³)", min_value=1000.0, value=RHO_WATER_DEFAULT, step=1.0, format="%.1f")
# Current
v_current_speed = st.sidebar.number_input("Mean Current Speed (v_c, m/s)", min_value=0.0, value=1.5, step=0.1, format="%.1f") # AB2018 uses 2 m/s for Hywind example

# Spar/Platform Parameters
st.sidebar.subheader("Spar/Platform Parameters")
D_spar = st.sidebar.number_input("Spar Diameter (D_P or D_S, m)", min_value=1.0, value=14.4, step=0.1, format="%.1f") # Hywind example D_S varies, using lower part dia
B_spar_draft = st.sidebar.number_input("Spar Draft (B, m)", min_value=10.0, value=85.0, step=1.0, format="%.1f")
C_D_spar_wave = st.sidebar.number_input("Spar Drag Coeff. for Waves (C_D_wave)", min_value=0.1, value=0.6, step=0.05, format="%.2f") # Typical for cylinder
C_m_spar_wave = st.sidebar.number_input("Spar Inertia Coeff. for Waves (C_m_wave)", min_value=0.5, value=2.0, step=0.1, format="%.1f") # Typical for cylinder
C_D_spar_current = st.sidebar.number_input("Spar Drag Coeff. for Current (C_D_current)", min_value=0.1, value=0.7, step=0.05, format="%.2f")

# Mooring & Anchor Parameters
st.sidebar.subheader("Mooring & Anchor Parameters")
soil_type = st.sidebar.selectbox("Soil Type", ["Clay", "Sand"], index=0)
# Forerunner
A_b_forerunner = st.sidebar.number_input("Forerunner Effective Bearing Area (A_b, m²)", min_value=0.01, value=0.3, step=0.01, format="%.2f") # e.g. chain dia * 2.5 * length_unit
Nc_forerunner_bearing = st.sidebar.number_input("Forerunner Bearing Capacity Factor (N_c)", min_value=1.0, value=9.0, step=0.5, format="%.1f")
mu_soil_friction_forerunner = st.sidebar.number_input("Forerunner-Soil Friction Coeff (μ_soil)", min_value=0.0, value=0.25, step=0.01, format="%.2f")
theta_mudline_deg_initial = st.sidebar.number_input("Initial Mooring Angle at Mudline (θ_m, degrees, for taut line approx.)", min_value=0.0, value=5.0, step=1.0, format="%.1f")

# Caisson Sizing
st.sidebar.subheader("Suction Caisson Sizing")
L_D_ratio_caisson = st.sidebar.number_input("Target Caisson Length-to-Diameter Ratio (L/D)", min_value=1.0, value=3.2, step=0.1, format="%.1f")
padeye_loc_ratio_z_a_L = st.sidebar.number_input("Padeye Location from Top (z_a/L)", min_value=0.1, value=0.6, step=0.05, format="%.2f") # e.g. 0.5 for clay, 2/3 for sand
t_wall_D_ratio_caisson = st.sidebar.number_input("Caisson Wall Thickness to Diameter Ratio (D/t_wall)", min_value=10.0, value=70.0, step=5.0, format="%.1f")
rho_steel_submerged_kg_m3 = st.sidebar.number_input("Submerged Steel Density (kg/m³)", value=6825.0, help="Approx (7850 - 1025)")


if soil_type == "Clay":
    s_u_profile_clay = st.sidebar.selectbox("Clay Undrained Shear Strength (s_u) Profile", ["Constant", "Linearly Increasing"], index=0)
    if s_u_profile_clay == "Constant":
        s_u_avg_clay = st.sidebar.number_input("Average s_u (kPa)", min_value=1.0, value=30.0, step=1.0, format="%.1f")
        s_u_tip_clay = s_u_avg_clay
        s_u0_clay = s_u_avg_clay
        k_su_increase_clay = 0.0
    else: # Linearly Increasing
        s_u0_clay = st.sidebar.number_input("s_u at Mudline (s_u0, kPa)", min_value=1.0, value=15.0, step=1.0, format="%.1f")
        k_su_increase_clay = st.sidebar.number_input("s_u Increase Rate (kPa/m)", min_value=0.0, value=2.0, step=0.1, format="%.1f")
        # s_u_avg and s_u_tip will be calculated based on caisson length L later
    
    N_p_lateral_clay = st.sidebar.number_input("Lateral Bearing Factor Clay (N_p)", min_value=1.0, value=10.0, step=0.5, format="%.1f") # AB2018 Table 2
    Nc_tip_clay = st.sidebar.number_input("Tip Bearing Factor Clay (N_c,tip)", min_value=1.0, value=9.0, step=0.5, format="%.1f")
    alpha_e_clay = st.sidebar.number_input("External Shaft Friction Coeff. Clay (α_e)", min_value=0.1, value=0.7, step=0.05, format="%.2f")
    alpha_i_clay = st.sidebar.number_input("Internal Shaft Friction Coeff. Clay (α_i)", min_value=0.1, value=0.5, step=0.05, format="%.2f")
    gamma_prime_soil_plug_clay = st.sidebar.number_input("Submerged Unit Weight of Soil Plug (γ'_plug, kN/m³)", min_value=1.0, value=8.0, step=0.5, format="%.1f")


elif soil_type == "Sand":
    phi_sand_deg = st.sidebar.number_input("Sand Internal Friction Angle (φ', degrees)", min_value=10.0, value=30.0, step=1.0, format="%.1f")
    gamma_prime_sand = st.sidebar.number_input("Sand Submerged Unit Weight (γ', kN/m³)", min_value=5.0, value=9.0, step=0.5, format="%.1f")
    K_tan_delta_ext_sand = st.sidebar.number_input("(K tan δ)_e for Sand", min_value=0.1, value=0.5, step=0.05, format="%.2f") # Typical values
    K_tan_delta_int_sand = st.sidebar.number_input("(K tan δ)_i for Sand", min_value=0.1, value=0.3, step=0.05, format="%.2f") # Typical values

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

        # Thrust Force (EOG at U_R) - F_u,EOG
        C_T_EOG = get_thrust_coefficient(U_R_turbine, U_R_turbine, C_T_method_choice, U_cut_in_turbine, U_cut_out_turbine) # At U_R, effective wind is U_R + u_EOG
        F_u_EOG = calculate_rotor_thrust_force(rho_air, D_rotor, C_T_EOG, U_R_turbine + u_EOG_val) / 1e6 # MN
        results['F_u_EOG_MN'] = F_u_EOG
        st.write(f"Rotor Thrust (EOG at U_R, F_u,EOG): {F_u_EOG:.2f} MN (C_T={C_T_EOG:.2f} at U_eff={U_R_turbine + u_EOG_val:.2f} m/s)")
        
        # Tower Drag (at U_R + u_EOG for this load case)
        F_DT_EOG = calculate_tower_drag_force(rho_air, C_DT_tower, U_R_turbine + u_EOG_val, z_hub, D_tower_bottom, D_tower_top, gamma_shear) / 1e6 # MN
        results['F_DT_EOG_MN'] = F_DT_EOG
        st.write(f"Tower Drag (at U_R + u_EOG, F_DT,EOG): {F_DT_EOG:.2f} MN")

        # Shutdown Thrust (50-yr extreme wind) - F_u,U50
        # Tower drag at U10_50yr
        F_DT_U50 = calculate_tower_drag_force(rho_air, C_DT_tower, U10_50yr, z_hub, D_tower_bottom, D_tower_top, gamma_shear) / 1e6 #MN
        results['F_DT_U50_MN'] = F_DT_U50
        st.write(f"Tower Drag (at U_10,50yr, F_DT,U50): {F_DT_U50:.2f} MN")
        
        F_u_U50_total = calculate_shutdown_thrust_drag(rho_air, U10_50yr, D_rotor, C_D_rotor_parked, F_DT_U50*1e6) / 1e6 # MN (pass F_DT in N)
        results['F_u_U50_total_MN'] = F_u_U50_total # This includes tower drag
        st.write(f"Total Shutdown Wind Load (Rotor Drag + Tower Drag at U_10,50yr, F_u,U50_total): {F_u_U50_total:.2f} MN")


    # --- 2. Wave Load Calculations ---
    with st.expander("2. Wave Load Calculations", expanded=True):
        H_S1, T_S1, H_m1, T_m1, H_S50, T_S50, H_m50, T_m50 = get_wave_design_parameters(H_S50_known)
        results.update({'H_S1':H_S1, 'T_S1':T_S1, 'H_m1':H_m1, 'T_m1':T_m1, 'H_S50':H_S50, 'T_S50':T_S50, 'H_m50':H_m50, 'T_m50':T_m50})
        st.write(f"1-yr Waves: H_S1={H_S1:.2f}m, T_S1={T_S1:.2f}s; H_m1={H_m1:.2f}m, T_m1={T_m1:.2f}s")
        st.write(f"50-yr Waves: H_S50={H_S50:.2f}m, T_S50={T_S50:.2f}s; H_m50={H_m50:.2f}m, T_m50={T_m50:.2f}s")

        # Wave loads for 1-yr EWH (H_m1, T_m1)
        omega_1yr = 2 * PI / T_m1 if T_m1 > 0 else 0
        k_1yr = solve_dispersion_relation_iterative(omega_1yr, S_water_depth)
        F_D_max_1yr, F_I_max_1yr = calculate_morison_wave_forces(rho_water, D_spar, C_D_spar_wave, C_m_spar_wave, H_m1, T_m1, S_water_depth, B_spar_draft, k_1yr)
        F_w_1yr_MN = (F_D_max_1yr + F_I_max_1yr) / 1e6 # MN
        results.update({'k_1yr':k_1yr, 'F_D_max_1yr_MN':F_D_max_1yr/1e6, 'F_I_max_1yr_MN':F_I_max_1yr/1e6, 'F_w_1yr_MN':F_w_1yr_MN})
        st.write(f"1-yr EWH Load (F_w,1yr): {F_w_1yr_MN:.2f} MN (Drag: {F_D_max_1yr/1e6:.2f} MN, Inertia: {F_I_max_1yr/1e6:.2f} MN, k={k_1yr:.4f} rad/m)")

        # Wave loads for 50-yr EWH (H_m50, T_m50)
        omega_50yr = 2 * PI / T_m50 if T_m50 > 0 else 0
        k_50yr = solve_dispersion_relation_iterative(omega_50yr, S_water_depth)
        F_D_max_50yr, F_I_max_50yr = calculate_morison_wave_forces(rho_water, D_spar, C_D_spar_wave, C_m_spar_wave, H_m50, T_m50, S_water_depth, B_spar_draft, k_50yr)
        F_w_50yr_MN = (F_D_max_50yr + F_I_max_50yr) / 1e6 # MN
        results.update({'k_50yr':k_50yr, 'F_D_max_50yr_MN':F_D_max_50yr/1e6, 'F_I_max_50yr_MN':F_I_max_50yr/1e6, 'F_w_50yr_MN':F_w_50yr_MN})
        st.write(f"50-yr EWH Load (F_w,50yr): {F_w_50yr_MN:.2f} MN (Drag: {F_D_max_50yr/1e6:.2f} MN, Inertia: {F_I_max_50yr/1e6:.2f} MN, k={k_50yr:.4f} rad/m)")

    # --- 3. Current Load Calculation ---
    with st.expander("3. Current Load Calculation", expanded=True):
        F_C_MN = calculate_current_force(rho_water, D_spar, C_D_spar_current, v_current_speed, B_spar_draft) / 1e6 # MN
        results['F_C_MN'] = F_C_MN
        st.write(f"Total Current Load (F_C): {F_C_MN:.2f} MN")

    # --- 4. ULS Load Combinations (at floater padeye) ---
    with st.expander("4. ULS Load Combinations (at floater padeye)", expanded=True):
        # E-1: 50-yr wind (shutdown) + 50-yr wave + current (AB2018 Eq 38)
        # F_u,U50_total already includes tower drag F_DT_U50
        F_E1_MN = results['F_u_U50_total_MN'] + results['F_w_50yr_MN'] + results['F_C_MN']
        results['F_E1_MN'] = F_E1_MN
        st.write(f"Load Case E-1 (50yr wind_shut + 50yr wave + current): {F_E1_MN:.2f} MN")

        # E-2: Max wind (EOG at U_R) + 1-yr wave + current (AB2018 Eq 39)
        # F_u_EOG + F_DT_EOG (rotor thrust + tower drag for EOG condition)
        F_wind_EOG_total_MN = results['F_u_EOG_MN'] + results['F_DT_EOG_MN']
        F_E2_MN = F_wind_EOG_total_MN + results['F_w_1yr_MN'] + results['F_C_MN']
        results['F_E2_MN'] = F_E2_MN
        st.write(f"Load Case E-2 (EOG@U_R wind + 1yr wave + current): {F_E2_MN:.2f} MN")
        
        F_ULS_floater_MN = max(F_E1_MN, F_E2_MN)
        results['F_ULS_floater_MN'] = F_ULS_floater_MN
        st.markdown(f"**Dominant ULS Horizontal Load at Floater Padeye: {F_ULS_floater_MN:.2f} MN**")
        T_mudline_N = F_ULS_floater_MN * 1e6 # For anchor calcs, use N

    # --- 5. Load Transfer to Anchor ---
    with st.expander("5. Load Transfer to Anchor", expanded=True):
        # Padeye depth z_a is calculated based on L_caisson later.
        # For now, Q_avg and padeye load will be calculated inside the iteration loop for caisson sizing.
        # Here, we can show a preliminary calculation if L is assumed or just state it's part of sizing.
        st.write("Anchor padeye load (T_a, θ_a) will be determined iteratively during caisson sizing.")
        st.write(f"Using ULS Floater Load as initial mudline tension T_m = {T_mudline_N/1e3:.0f} kN at θ_m = {theta_mudline_deg_initial:.1f}°")


    # --- 6. Suction Caisson Sizing (Iterative) ---
    with st.expander("6. Suction Caisson Sizing", expanded=True):
        st.write(f"Target L/D = {L_D_ratio_caisson:.2f}, Padeye z_a/L = {padeye_loc_ratio_z_a_L:.2f}")
        
        # Iterative search for minimum D_caisson
        D_caisson_current = 2.0 # Initial guess for Diameter in m
        D_increment = 0.1 # m
        max_D_caisson = 20.0 # Max search limit
        found_D = False
        
        best_D_so_far = -1
        best_FP_so_far = float('inf')

        iterations_data = []

        for i in range(int((max_D_caisson - D_caisson_current)/D_increment) + 1):
            L_caisson_current = D_caisson_current * L_D_ratio_caisson
            z_a_padeye_current = padeye_loc_ratio_z_a_L * L_caisson_current

            # Calculate s_u_avg and s_u_tip if clay is linearly increasing
            current_s_u_avg_clay = 0
            current_s_u_tip_clay = 0
            if soil_type == "Clay":
                if s_u_profile_clay == "Constant":
                    current_s_u_avg_clay = s_u_avg_clay
                    current_s_u_tip_clay = s_u_tip_clay
                else: # Linearly Increasing
                    current_s_u_avg_clay = s_u0_clay + k_su_increase_clay * (L_caisson_current / 2) # Avg over length L
                    current_s_u_tip_clay = s_u0_clay + k_su_increase_clay * L_caisson_current       # At tip L
            
            gamma_prime_for_Qavg = gamma_prime_sand * 1000 if soil_type == "Sand" else 0 # N/m3
            s_u_for_Qavg = current_s_u_avg_clay * 1000 if soil_type == "Clay" else 0 # Pa (using avg s_u over embedment for forerunner)
            
            Q_avg_val = calculate_Q_avg_forerunner(soil_type, s_u_for_Qavg, gamma_prime_for_Qavg, z_a_padeye_current, A_b_forerunner, Nc_forerunner_bearing)
            
            T_a_N, theta_a_deg = solve_anchor_padeye_load_system(T_mudline_N, theta_mudline_deg_initial, z_a_padeye_current, Q_avg_val, mu_soil_friction_forerunner)
            
            H_u_anchor_N = T_a_N * math.cos(to_radians(theta_a_deg))
            V_u_anchor_N = T_a_N * math.sin(to_radians(theta_a_deg))
            if V_u_anchor_N < 0: V_u_anchor_N = 0 # No uplift resistance for negative V_u

            W_prime_caisson_N, W_prime_plug_N, t_wall_m, D_i_m = estimate_caisson_weights(
                D_caisson_current, L_caisson_current, t_wall_D_ratio_caisson, rho_steel_submerged_kg_m3, 
                (gamma_prime_soil_plug_clay if soil_type == "Clay" else gamma_prime_sand) * 1000, # kN/m3 to N/m3
                soil_type
            )

            H_m_cap_N, V_m_cap_N = 0, 0
            debug_cap_info = {}
            if soil_type == "Clay":
                H_m_cap_N, V_m_cap_N, F_ext_f, F_int_f, F_rev_eb, Vm1, Vm2, Vm3 = calculate_caisson_capacity_clay(
                    L_caisson_current, D_caisson_current, D_i_m, current_s_u_avg_clay, current_s_u_tip_clay,
                    N_p_lateral_clay, Nc_tip_clay, alpha_e_clay, alpha_i_clay,
                    W_prime_caisson_N, W_prime_plug_N
                )
                debug_cap_info = {'Vm1_kN':Vm1/1e3, 'Vm2_kN':Vm2/1e3, 'Vm3_kN':Vm3/1e3, 'F_ext_f_kN':F_ext_f/1e3, 'F_int_f_kN':F_int_f/1e3, 'F_rev_eb_kN':F_rev_eb/1e3}
            elif soil_type == "Sand":
                N_q_s = calculate_Nq_sand(phi_sand_deg)
                H_m_cap_N, V_m_cap_N = calculate_caisson_capacity_sand(
                    L_caisson_current, D_caisson_current, D_i_m, gamma_prime_sand * 1000, N_q_s, # N/m3
                    W_prime_caisson_N, K_tan_delta_ext_sand, K_tan_delta_int_sand
                )
                debug_cap_info = {'Nq_sand': N_q_s}

            FP_val, a_exp, b_exp = check_caisson_failure_criterion(H_u_anchor_N, V_u_anchor_N, H_m_cap_N, V_m_cap_N, L_caisson_current, D_caisson_current)
            
            iterations_data.append({
                "D (m)": D_caisson_current, "L (m)": L_caisson_current, "z_a (m)": z_a_padeye_current,
                "T_a (kN)": T_a_N/1e3, "θ_a (deg)": theta_a_deg,
                "H_u (kN)": H_u_anchor_N/1e3, "V_u (kN)": V_u_anchor_N/1e3,
                "H_m_cap (kN)": H_m_cap_N/1e3, "V_m_cap (kN)": V_m_cap_N/1e3,
                "FP": FP_val, "a": a_exp, "b": b_exp,
                "W'_c (kN)": W_prime_caisson_N/1e3, "W'_plug (kN)": W_prime_plug_N/1e3,
                **debug_cap_info
            })

            if FP_val <= 1.0:
                if not found_D or D_caisson_current < best_D_so_far : # Found a valid D, or a smaller valid D
                    best_D_so_far = D_caisson_current
                    results['D_caisson_final_m'] = D_caisson_current
                    results['L_caisson_final_m'] = L_caisson_current
                    results['t_wall_final_m'] = t_wall_m
                    results['z_a_padeye_final_m'] = z_a_padeye_current
                    results['T_a_final_kN'] = T_a_N / 1e3
                    results['theta_a_final_deg'] = theta_a_deg
                    results['H_u_anchor_final_kN'] = H_u_anchor_N / 1e3
                    results['V_u_anchor_final_kN'] = V_u_anchor_N / 1e3
                    results['H_m_cap_final_kN'] = H_m_cap_N / 1e3
                    results['V_m_cap_final_kN'] = V_m_cap_N / 1e3
                    results['FP_final'] = FP_val
                    results['a_exp_final'] = a_exp
                    results['b_exp_final'] = b_exp
                    results['W_prime_caisson_final_kN'] = W_prime_caisson_N / 1e3
                    results['W_prime_plug_final_kN'] = W_prime_plug_N / 1e3
                    if soil_type == "Clay": results.update(debug_cap_info)
                    if soil_type == "Sand": results.update(debug_cap_info)
                    found_D = True
                    break # Stop at first D that satisfies FP <= 1 (smallest D due to incrementing search)
            
            D_caisson_current += D_increment

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
            if soil_type == "Clay":
                 st.write(f"Clay Vm details (kN): Vm1={results.get('Vm1_kN',0):.1f}, Vm2={results.get('Vm2_kN',0):.1f}, Vm3={results.get('Vm3_kN',0):.1f} (Min chosen)")
            if soil_type == "Sand" and 'Nq_sand' in results:
                 st.write(f"Sand Nq factor: {results.get('Nq_sand',0):.2f}")

        else:
            st.error(f"Could not find suitable caisson diameter up to {max_D_caisson:.1f} m for L/D={L_D_ratio_caisson:.1f}. Last FP = {FP_val:.3f}")

        if iterations_data:
            st.subheader("Sizing Iteration Details")
            # Select key columns for display to avoid overly wide table
            display_cols = ["D (m)", "L (m)", "H_u (kN)", "V_u (kN)", "H_m_cap (kN)", "V_m_cap (kN)", "FP"]
            df_iterations = st.dataframe([{k: v for k,v in row.items() if k in display_cols} for row in iterations_data], height=300)


    # --- Disclaimer ---
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Disclaimer:** This tool is for educational and preliminary estimation purposes only. 
    Results should be verified by qualified professionals using detailed engineering analysis and relevant standards.
    """)

else:
    st.info("Adjust parameters in the sidebar and click 'Run Analysis'.")
