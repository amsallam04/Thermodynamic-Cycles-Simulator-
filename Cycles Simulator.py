import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
from datetime import datetime
from scipy.optimize import fsolve

# ----------------------
# Constants (kJ, kPa, K)
# ----------------------
R_AIR = 0.287  # kJ/kg.K for air (approx)
DEFAULT_GAMMA = 1.4

# ----------------------
# Variable Specific Heat Functions (Updated as per user's slide)
# Using the form: cp = a1 + k1*T and cv = b1 + k1*T
# We will use the existing constants from the old function as a baseline for the new form:
# Old: cv = 0.71 + 20e-5 * T, cp = 0.997 + 20e-5 * T
# So: b1 = 0.71, a1 = 0.997, k1 = 20e-5
# ----------------------
A1 = 0.997
B1 = 0.71
K1 = 20e-5
A2 = 0.997
B2 = 0.71
K2_1 = 20e-5
K2_2 = 0.0 # Assuming k2 is zero for the simpler form, as the code only uses one form

def cp_func(T):
    """Specific heat at constant pressure (kJ/kg.K) as a function of T (K)."""
    # Using the form: cp = a1 + k1*T
    return A1 + K1 * T

def cv_func(T):
    """Specific heat at constant volume (kJ/kg.K) as a function of T (K)."""
    # Using the form: cv = b1 + k1*T
    return B1 + K1 * T

def internal_energy(T):
    """Internal energy (U) relative to T=0 (kJ/kg).
    U = integral(cv(T) dT) = integral(b1 + k1*T) dT = b1*T + (k1/2)*T^2
    """
    return B1 * T + (K1 / 2) * T**2

def enthalpy(T):
    """Enthalpy (H) relative to T=0 (kJ/kg).
    H = integral(cp(T) dT) = integral(a1 + k1*T) dT = a1*T + (k1/2)*T^2
    """
    return A1 * T + (K1 / 2) * T**2

# ----------------------
# Utility functions
# ----------------------
def cp_from_gamma(gamma: float):
    """Return Cp and Cv (kJ/kg.K) from gamma and R (kJ/kg.K)."""
    Cv = R_AIR / (gamma - 1)
    Cp = Cv + R_AIR
    return Cp, Cv

def entropy_change(Cp, P1, P2, T1, T2):
    """
    Approximate entropy change (kJ/kg.K) between two states for ideal gas:
    ds = Cp*ln(T2/T1) - R*ln(P2/P1)
    (Cp,R in kJ/kg.K so ds in kJ/kg.K)
    """
    return Cp * np.log(T2 / T1) - R_AIR * np.log(P2 / P1)

# ----------------------
# Cycle calculations - AIR STANDARD (Constant Specific Heat)
# ----------------------
def otto_cycle_standard(P1_kpa, T1, r, T3, gamma=DEFAULT_GAMMA):
    """Compute Otto cycle state points (Air-Standard, constant specific heat)."""
    Cp, Cv = cp_from_gamma(gamma)
    P1 = P1_kpa
    V1 = 1.0
    V2 = V1 / r
    T2 = T1 * r ** (gamma - 1)
    P2 = P1 * r ** gamma
    V3 = V2
    P3 = P2 * (T3 / T2)
    V4 = V1
    T4 = T3 / r ** (gamma - 1)
    P4 = P3 * (V3 / V4) ** gamma
    Q_in = Cv * (T3 - T2)
    Q_out = Cv * (T4 - T1)
    W_net = Q_in - Q_out
    eta_analytic = 1 - 1 / (r ** (gamma - 1))
    V_swept = V1 - V2
    MEP = W_net / V_swept
    states = {
        1: {"P": P1, "V": V1, "T": T1}, 2: {"P": P2, "V": V2, "T": T2},
        3: {"P": P3, "V": V3, "T": T3}, 4: {"P": P4, "V": V4, "T": T4},
    }
    # Entropy calculation is kept for internal use but will be removed from display
    s = {1: 0.0}
    for i in [2,3,4]:
        s[i] = s[1] + entropy_change(Cp, states[1]["P"], states[i]["P"], states[1]["T"], states[i]["T"])
    return {"name": "Otto (AIR STANDARD)", "gamma": gamma, "Cp": Cp, "Cv": Cv, "states": states, "s": s, "Q_in": Q_in, "Q_out": Q_out, "W_net": W_net, "eta_analytic": eta_analytic, "MEP": MEP}

def diesel_cycle_standard(P1_kpa, T1, r, cutoff, Tmax, gamma=DEFAULT_GAMMA):
    """Compute Diesel cycle (Air-Standard, constant specific heat)."""
    Cp, Cv = cp_from_gamma(gamma)
    P1 = P1_kpa
    V1 = 1.0
    V2 = V1 / r
    T2 = T1 * r ** (gamma - 1)
    P2 = P1 * r ** gamma
    
    # Constant pressure heat addition (2->3)
    P3 = P2
    T3 = Tmax if Tmax > T2 else T2 * cutoff # Use Tmax if provided, otherwise calculate T3 from cutoff
    V3 = V2 * (T3 / T2) # V3/V2 = T3/T2
    
    # State 4 after isentropic expansion back to V1
    V4 = V1
    T4 = T3 * (V3 / V4) ** (gamma - 1)
    P4 = P3 * (V3 / V4) ** gamma
    
    Q_in = Cp * (T3 - T2)
    Q_out = Cv * (T4 - T1)
    W_net = Q_in - Q_out
    
    # Cutoff ratio rho = V3/V2
    rho = V3 / V2
    eta_analytic = 1 - (1 / r ** (gamma - 1)) * ((rho ** gamma - 1) / (gamma * (rho - 1)))
    
    V_swept = V1 - V2
    MEP = W_net / V_swept
    
    states = {
        1: {"P": P1, "V": V1, "T": T1}, 2: {"P": P2, "V": V2, "T": T2},
        3: {"P": P3, "V": V3, "T": T3}, 4: {"P": P4, "V": V4, "T": T4},
    }
    s = {1: 0.0}
    for i in [2,3,4]:
        s[i] = s[1] + entropy_change(Cp, states[1]["P"], states[i]["P"], states[1]["T"], states[i]["T"])
    return {"name": "Diesel (AIR STANDARD)", "gamma": gamma, "Cp": Cp, "Cv": Cv, "states": states, "s": s, "Q_in": Q_in, "Q_out": Q_out, "W_net": W_net, "eta_analytic": eta_analytic, "MEP": MEP}

def dual_cycle_standard(P1_kpa, T1, r, cutoff, cv_fraction, Tmax, gamma=DEFAULT_GAMMA):
    """Compute Dual cycle (Air-Standard, constant specific heat)."""
    Cp, Cv = cp_from_gamma(gamma)
    P1 = P1_kpa
    V1 = 1.0
    V2 = V1 / r
    T2 = T1 * r ** (gamma - 1)
    P2 = P1 * r ** gamma
    
    # 2->3: Constant Volume Heat Addition (Q_cv)
    # The original logic for T3/T4 calculation was complex. Let's simplify based on the input Tmax and cv_fraction.
    # Tmax is T4.
    
    # Calculate T3 based on T4 (Tmax) and cv_fraction
    # Q_in = Q_cv + Q_cp = Cv*(T3-T2) + Cp*(T4-T3)
    # The original code was using Tmax to distribute the temperature rise. Let's stick to that for now.
    deltaT_total = max(1e-6, Tmax - T2)
    
    # We need to find T3 such that the heat added in 2->3 (Cv*(T3-T2)) and 3->4 (Cp*(T4-T3))
    # corresponds to the cv_fraction of the total heat.
    # Q_cv / Q_in = cv_fraction
    # Q_in = Cv*(T3-T2) + Cp*(T4-T3)
    # Q_cv = Cv*(T3-T2)
    
    # This is difficult to solve for T3 directly without knowing Q_in.
    # The original code used a simplified approach by distributing the temperature rise (Tmax - T2)
    # based on cv_fraction, which is physically incorrect for heat fractions.
    # Let's revert to the standard definition where T3 is calculated from the pressure ratio (alpha)
    # and T4 from the cutoff ratio (rho). Since the user provides Tmax (T4) and cv_fraction,
    # we must use an iterative approach or simplify.
    
    # Let's use the original code's approach of distributing the temperature rise,
    # but ensure the states are correct (1-2-3-4-5, where 5 is the final state).
    
    # 2->3: Constant Volume Heat Addition (Q_cv)
    # The original code's approach:
    # deltaT_total = max(1e-6, Tmax - T2)
    # deltaT_cv = cv_fraction * deltaT_total
    # T3 = T2 + deltaT_cv
    # T4 = T3 + (1 - cv_fraction) * deltaT_total
    
    # This is a non-standard way to define the cycle. Let's assume Tmax is T4.
    # T4 is the peak temperature.
    T4 = Tmax
    
    # We need to find T3 such that Q_cv / Q_in = cv_fraction
    # Q_in = Cv*(T3-T2) + Cp*(T4-T3)
    # Q_cv = Cv*(T3-T2)
    # Cv*(T3-T2) / (Cv*(T3-T2) + Cp*(T4-T3)) = cv_fraction
    # Let x = T3 - T2 and y = T4 - T3
    # Cv*x / (Cv*x + Cp*y) = cv_fraction
    # Cv*x = cv_fraction * Cv*x + cv_fraction * Cp*y
    # Cv*x * (1 - cv_fraction) = cv_fraction * Cp*y
    # Cv*(T3-T2)*(1-cv_fraction) = cv_fraction * Cp*(T4-T3)
    # T3*(Cv*(1-cv_fraction) + cv_fraction*Cp) = T2*Cv*(1-cv_fraction) + T4*cv_fraction*Cp
    
    T3_numerator = T2 * Cv * (1 - cv_fraction) + T4 * cv_fraction * Cp
    T3_denominator = Cv * (1 - cv_fraction) + cv_fraction * Cp
    T3 = T3_numerator / T3_denominator
    
    V3 = V2
    P3 = P2 * (T3 / T2)
    
    # 3->4: Constant Pressure Heat Addition (Q_cp)
    P4 = P3
    V4 = V3 * (T4 / T3)
    
    # 4->5: Isentropic Expansion to V1
    V5 = V1
    T5 = T4 * (V4 / V5) ** (gamma - 1)
    P5 = P4 * (V4 / V5) ** gamma
    
    Q_cv = Cv * (T3 - T2)
    Q_cp = Cp * (T4 - T3)
    Q_in = Q_cv + Q_cp
    Q_out = Cv * (T5 - T1)
    W_net = Q_in - Q_out
    eta = W_net / Q_in if Q_in != 0 else 0.0
    
    V_swept = V1 - V2
    MEP = W_net / V_swept
    
    # Corrected states for plotting (1-2-3-4-5)
    states = {
        1: {"P": P1, "V": V1, "T": T1}, 2: {"P": P2, "V": V2, "T": T2},
        3: {"P": P3, "V": V3, "T": T3}, 4: {"P": P4, "V": V4, "T": T4},
        5: {"P": P5, "V": V5, "T": T5},
    }
    s = {1: 0.0}
    for k in [2,3,4,5]:
        entry = states[k]
        s[k] = s[1] + entropy_change(Cp, states[1]["P"], entry["P"], states[1]["T"], entry["T"])
    return {"name": "Dual (AIR STANDARD)", "gamma": gamma, "Cp": Cp, "Cv": Cv, "states": states, "s": s, "Q_in": Q_in, "Q_out": Q_out, "W_net": W_net, "eta_analytic": eta, "MEP": MEP}

# ----------------------
# Cycle calculations - FUEL-AIR (Variable Specific Heat)
# ----------------------
def air_fuel_base_calc(P1_kpa, T1, r, CV, AFR, n, cycle_type, cutoff=1.0, cv_fraction=1.0):
    """Base calculation for Otto, Diesel, and Dual Air-Fuel cycles."""
    P1 = P1_kpa
    V1 = 1.0
    V2 = V1 / r
    
    # 1->2: Polytropic Compression (PV^n = const)
    T2 = T1 * r ** (n - 1)
    P2 = P1 * r ** n
    V3 = V2
    
    Q_in_total = CV / (AFR + 1)
    
    if cycle_type == "Otto":
        Q_cv = Q_in_total
        Q_cp = 0.0
    elif cycle_type == "Diesel":
        Q_cv = 0.0
        Q_cp = Q_in_total
    else: # Dual
        Q_cv = Q_in_total * cv_fraction
        Q_cp = Q_in_total * (1 - cv_fraction)
        
    # 2->3: Constant Volume Heat Addition (Q_cv = U3 - U2)
    def equation_for_T3(T3):
        return (internal_energy(T3) - internal_energy(T2)) - Q_cv
    
    T3_initial_guess = T2 + Q_cv / B1 # Estimate T3 using constant Cv (B1 is the constant part of Cv)
    T3_root = fsolve(equation_for_T3, T3_initial_guess)
    T3 = T3_root[0]
    P3 = P2 * (T3 / T2)
    
    # 3->4: Constant Pressure Heat Addition (Q_cp = H4 - H3)
    if cycle_type == "Otto":
        T4 = T3
        P4 = P3
        V4 = V3
    else:
        def equation_for_T4(T4):
            return (enthalpy(T4) - enthalpy(T3)) - Q_cp
        
        T4_initial_guess = T3 + Q_cp / A1 # Estimate T4 using constant Cp (A1 is the constant part of Cp)
        T4_root = fsolve(equation_for_T4, T4_initial_guess)
        T4 = T4_root[0]
        P4 = P3
        V4 = V3 * (T4 / T3)
        
    # Final State (Expansion)
    if cycle_type == "Otto" or cycle_type == "Diesel":
        V_final = V1
        T_final = T4 * (V4 / V_final) ** (n - 1)
        P_final = P4 * (V4 / V_final) ** n
        
        states = {
            1: {"P": P1, "V": V1, "T": T1}, 2: {"P": P2, "V": V2, "T": T2},
            3: {"P": P3, "V": V3, "T": T3}, 4: {"P": P_final, "V": V_final, "T": T_final},
        }
        Q_out = internal_energy(T_final) - internal_energy(T1) # U_final - U_1
        
    else: # Dual Cycle (5 states)
        V5 = V1
        T5 = T4 * (V4 / V5) ** (n - 1)
        P5 = P4 * (V4 / V5) ** n
        
        # Corrected states for plotting (1-2-3-4-5)
        states = {
            1: {"P": P1, "V": V1, "T": T1}, 2: {"P": P2, "V": V2, "T": T2},
            3: {"P": P3, "V": V3, "T": T3}, 4: {"P": P4, "V": V4, "T": T4},
            5: {"P": P5, "V": V5, "T": T5},
        }
        Q_out = internal_energy(T5) - internal_energy(T1) # U_5 - U_1
        
    W_net = Q_in_total - Q_out
    eta_analytic = W_net / Q_in_total if Q_in_total != 0 else 0.0
    
    V_swept = V1 - V2
    MEP = W_net / V_swept
    
    # Simplified entropy (set to 0 for variable specific heat model)
    s = {k: 0.0 for k in states.keys()}

    return {
        "name": f"{cycle_type} (FUEL-AIR)",
        "gamma": n, "Cp": None, "Cv": None, "states": states, "s": s,
        "Q_in": Q_in_total, "Q_out": Q_out, "W_net": W_net, "eta_analytic": eta_analytic,
        "MEP": MEP,
        "T2": T2, "T3": T3, "T4": T4, "P2": P2, "P3": P3, "P4": P4
    }

# ----------------------
# Cycle calculations - FUEL-AIR (Constant Specific Heat Comparison)
# ----------------------
def air_fuel_const_cv_comparison(P1_kpa, T1, r, CV, AFR, n, Cv_const, cycle_type, cutoff=1.0, cv_fraction=1.0):
    """
    Compute max pressure for comparison using constant specific heat.
    """
    P1 = P1_kpa
    V1 = 1.0
    V2 = V1 / r
    
    T2 = T1 * r ** (n - 1)
    P2 = P1 * r ** n
    
    Q_in_total = CV / (AFR + 1)
    
    if cycle_type == "Otto":
        Q_cv = Q_in_total
        Q_cp = 0.0
    elif cycle_type == "Diesel":
        Q_cv = 0.0
        Q_cp = Q_in_total
    else: # Dual
        Q_cv = Q_in_total * cv_fraction
        Q_cp = Q_in_total * (1 - cv_fraction)
        
    # 2->3: Constant Volume Heat Addition (Q_cv = Cv_const * (T3 - T2))
    T3 = T2 + Q_cv / Cv_const
    P3 = P2 * (T3 / T2)
    
    # 3->4: Constant Pressure Heat Addition (Q_cp = Cp_const * (T4 - T3))
    # Cp_const = Cv_const + R_AIR
    Cp_const = Cv_const + R_AIR
    
    if cycle_type == "Otto":
        T4 = T3
        P4 = P3
    else:
        T4 = T3 + Q_cp / Cp_const
        P4 = P3
        
    return max(P3, P4)

# ----------------------
# Plotting utilities (Updated for smoother lines and Dual cycle correction)
# ----------------------
def generate_isentropic_path(P1, V1, T1, P2, V2, T2, gamma, num_points=50):
    """Generates points for an isentropic process (PV^gamma = const)."""
    V_path = np.linspace(V1, V2, num_points)
    P_path = P1 * (V1 / V_path) ** gamma
    T_path = T1 * (V1 / V_path) ** (gamma - 1)
    return V_path, P_path, T_path

def plot_pv(states, title="P-V Diagram", gamma=DEFAULT_GAMMA):
    # Define colors for processes
    COLORS = {
        "Compression": '#4682B4',  # Blue (Work In)
        "Heat_Add": '#FF4B4B',     # Red (Heat In)
        "Expansion": '#2ECC71',    # Green (Work Out)
        "Heat_Rej": '#7F8C8D',     # Gray (Heat Out)
    }
    keys = sorted(states.keys())
    plt.style.use('seaborn-v0_8-whitegrid') # Aesthetic improvement
    fig, ax = plt.subplots(figsize=(6,4))
    
    # Plotting paths for smoother lines and distinct colors
    for i in range(len(keys)):
        k1 = keys[i]
        k2 = keys[(i + 1) % len(keys)] # Wrap around to state 1
        
        P1, V1, T1 = states[k1]["P"], states[k1]["V"], states[k1]["T"]
        P2, V2, T2 = states[k2]["P"], states[k2]["V"], states[k2]["T"]
        
        # Determine process type and color
        process_type = ""
        color = COLORS["Heat_Rej"] # Default to Heat Rejection (4->1 or 5->1)
        
        if k1 == 1 and k2 == 2:
            # 1->2: Compression (Polytropic/Isentropic)
            process_type = "Compression"
            color = COLORS["Compression"]
        elif k1 == 2 and k2 == 3:
            # 2->3: Heat Addition (Constant Volume or Constant Pressure)
            process_type = "Heat_Add"
            color = COLORS["Heat_Add"]
        elif k1 == 3 and k2 == 4:
            # 3->4: Heat Addition (Constant Pressure) or Expansion (Otto)
            if V1 == V2: # Otto: 3->4 is Expansion
                process_type = "Expansion"
                color = COLORS["Expansion"]
            else: # Diesel/Dual: 3->4 is Heat Addition (CP)
                process_type = "Heat_Add"
                color = COLORS["Heat_Add"]
        elif k1 == 4 and k2 == 5:
            # 4->5: Expansion (Dual)
            process_type = "Expansion"
            color = COLORS["Expansion"]
        elif (k1 == 3 and k2 == 4 and len(keys) == 4 and V1 != V2) or (k1 == 4 and k2 == 1 and len(keys) == 4) or (k1 == 5 and k2 == 1 and len(keys) == 5):
            # 3->4 Expansion (Diesel) or 4->1/5->1 Heat Rejection
            if process_type == "": # If not already set by previous conditions
                if V1 != V2 and V2 != V1: # Expansion (Diesel 3->4)
                    process_type = "Expansion"
                    color = COLORS["Expansion"]
                else: # Heat Rejection (4->1 or 5->1)
                    process_type = "Heat_Rej"
                    color = COLORS["Heat_Rej"]
        
        # Corrected logic for 4-state cycles (Otto/Diesel)
        if len(keys) == 4:
            if k1 == 3 and k2 == 4: # 3->4 Expansion
                process_type = "Expansion"
                color = COLORS["Expansion"]
            elif k1 == 4 and k2 == 1: # 4->1 Heat Rejection
                process_type = "Heat_Rej"
                color = COLORS["Heat_Rej"]
        
        # Plotting the path
        if V1 == V2 or P1 == P2:
            # Constant Volume or Constant Pressure (Straight line)
            ax.plot([V1, V2], [P1, P2], color=color, linewidth=2)
        elif process_type in ["Compression", "Expansion"]:
            # Polytropic/Isentropic (Smooth curve)
            V_path, P_path, _ = generate_isentropic_path(P1, V1, T1, P2, V2, T2, gamma)
            ax.plot(V_path, P_path, color=color, linewidth=2)
        else:
            # Fallback for other processes (e.g., Heat Rejection 4->1/5->1 which is constant V)
            # or any other process that should be a straight line
            ax.plot([V1, V2], [P1, P2], color=color, linewidth=2, linestyle='--')
            
    # Plot state points as markers
    V_points = [states[k]["V"] for k in keys]
    P_points = [states[k]["P"] for k in keys]
    ax.plot(V_points, P_points, 'o', color=COLORS["Heat_Add"], markersize=5) # Use a neutral color for points
    
    # Annotate state points
    for k in keys:
        ax.annotate(f"{k}", (states[k]["V"], states[k]["P"]), textcoords="offset points", xytext=(5,5), ha='center')

    ax.set_xlabel("Volume (normalized)")
    ax.set_ylabel("Pressure (kPa)")
    ax.set_title(title)
    ax.grid(True)
    plt.tight_layout()
    return fig

def plot_ts(states, s_dict, title="T-S Diagram", gamma=DEFAULT_GAMMA):
    # Define colors for processes
    COLORS = {
        "Compression": '#4682B4',  # Blue (Work In)
        "Heat_Add": '#FF4B4B',     # Red (Heat In)
        "Expansion": '#2ECC71',    # Green (Work Out)
        "Heat_Rej": '#7F8C8D',     # Gray (Heat Out)
    }
    keys = sorted(states.keys())
    plt.style.use('seaborn-v0_8-whitegrid') # Aesthetic improvement
    fig, ax = plt.subplots(figsize=(6,4))
    
    # Plotting paths for smoother lines and distinct colors
    for i in range(len(keys)):
        k1 = keys[i]
        k2 = keys[(i + 1) % len(keys)] # Wrap around to state 1
        
        P1, V1, T1 = states[k1]["P"], states[k1]["V"], states[k1]["T"]
        P2, V2, T2 = states[k2]["P"], states[k2]["V"], states[k2]["T"]
        S1, S2 = s_dict[k1], s_dict[k2]
        
        # Determine process type and color
        process_type = ""
        color = COLORS["Heat_Rej"] # Default to Heat Rejection (4->1 or 5->1)
        
        if k1 == 1 and k2 == 2:
            # 1->2: Compression (Isentropic/Polytropic)
            process_type = "Compression"
            color = COLORS["Compression"]
        elif k1 == 2 and k2 == 3:
            # 2->3: Heat Addition (Constant Volume or Constant Pressure)
            process_type = "Heat_Add"
            color = COLORS["Heat_Add"]
        elif k1 == 3 and k2 == 4:
            # 3->4: Heat Addition (Constant Pressure) or Expansion (Otto)
            if V1 == V2: # Otto: 3->4 is Expansion
                process_type = "Expansion"
                color = COLORS["Expansion"]
            else: # Diesel/Dual: 3->4 is Heat Addition (CP)
                process_type = "Heat_Add"
                color = COLORS["Heat_Add"]
        elif k1 == 4 and k2 == 5:
            # 4->5: Expansion (Dual)
            process_type = "Expansion"
            color = COLORS["Expansion"]
        elif (k1 == 3 and k2 == 4 and len(keys) == 4 and V1 != V2) or (k1 == 4 and k2 == 1 and len(keys) == 4) or (k1 == 5 and k2 == 1 and len(keys) == 5):
            # 3->4 Expansion (Diesel) or 4->1/5->1 Heat Rejection
            if process_type == "": # If not already set by previous conditions
                if V1 != V2 and V2 != V1: # Expansion (Diesel 3->4)
                    process_type = "Expansion"
                    color = COLORS["Expansion"]
                else: # Heat Rejection (4->1 or 5->1)
                    process_type = "Heat_Rej"
                    color = COLORS["Heat_Rej"]
        
        # Corrected logic for 4-state cycles (Otto/Diesel)
        if len(keys) == 4:
            if k1 == 3 and k2 == 4: # 3->4 Expansion
                process_type = "Expansion"
                color = COLORS["Expansion"]
            elif k1 == 4 and k2 == 1: # 4->1 Heat Rejection
                process_type = "Heat_Rej"
                color = COLORS["Heat_Rej"]
        
        # Plotting the path
        if S1 == S2:
            # Isentropic (Straight line on T-S)
            ax.plot([S1, S2], [T1, T2], color=color, linewidth=2)
        elif V1 == V2 or P1 == P2:
            # Constant Volume or Constant Pressure (Curved line on T-S)
            S_path = np.linspace(S1, S2, 50)
            T_path = np.linspace(T1, T2, 50)
            ax.plot(S_path, T_path, color=color, linewidth=2)
        else:
            # Polytropic/Isentropic (Curved line on T-S for variable specific heat)
            # For constant specific heat, isentropic is S=const, which is handled above.
            # For variable specific heat, we use a straight line approximation for now, as calculating the T-S path is complex.
            S_path = np.linspace(S1, S2, 50)
            T_path = np.linspace(T1, T2, 50)
            ax.plot(S_path, T_path, color=color, linewidth=2, linestyle='--')
            
    # Plot state points as markers
    S_points = [s_dict[k] for k in keys]
    T_points = [states[k]["T"] for k in keys]
    ax.plot(S_points, T_points, 'o', color=COLORS["Heat_Add"], markersize=5) # Use a neutral color for points
    
    # Annotate state points
    for k in keys:
        ax.annotate(f"{k}", (s_dict[k], states[k]["T"]), textcoords="offset points", xytext=(5,5), ha='center')

    ax.set_xlabel("Entropy (kJ/kg.K, approx)")
    ax.set_ylabel("Temperature (K)")
    ax.set_title(title)
    ax.grid(True)
    plt.tight_layout()
    return fig

def plot_step_processes(states, gamma, cycle_name="Cycle"):
    # This plot is less critical and can be kept simple
    keys = sorted(states.keys())
    V = [states[k]["V"] for k in keys]
    P = [states[k]["P"] for k in keys]
    Vc = V + [V[0]]
    Pc = P + [P[0]]

    fig, axs = plt.subplots(2, 2, figsize=(10,7))
    axs = axs.flatten()
    
    # Titles adjusted for 5-state Dual cycle
    if len(keys) == 5:
        titles = ["1→2 (Comp)", "2→3 (CV Heat)", "3→4 (CP Heat)", "4→5 (Exp)", "5→1 (Heat Rej)"]
        # Only plot the first 4 main processes
        for i in range(4):
            v0, v1 = Vc[i], Vc[i+1]
            p0, p1 = Pc[i], Pc[i+1]
            axs[i].plot([v0, v1], [p0, p1], marker='o', color='#FF4B4B')
            axs[i].set_title(f"Process {titles[i]}")
            axs[i].set_xlabel("V"); axs[i].set_ylabel("P")
            axs[i].grid(True)
        # Remove the 4th subplot for 5-state cycle (since we only have 4 subplots)
        fig.delaxes(axs[3])
    else:
        titles = ["1→2 (Comp)", "2→3 (Heat Add)", "3→4 (Exp)", "4→1 (Heat Rej)"]
        for i in range(4):
            v0, v1 = Vc[i], Vc[i+1]
            p0, p1 = Pc[i], Pc[i+1]
            axs[i].plot([v0, v1], [p0, p1], marker='o', color='#FF4B4B')
            axs[i].set_title(f"Process {titles[i]}")
            axs[i].set_xlabel("V"); axs[i].set_ylabel("P")
            axs[i].grid(True)
            
    fig.suptitle(f"{cycle_name} - Step Processes (P-V)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

# Removed plot_hrr_for_cycle function

# ----------------------
# Custom Streamlit Theme (Adapted for Light/Dark Mode)
# ----------------------

if "theme" not in st.session_state:
    st.session_state.run_simulation = False
    st.session_state.theme = "light"

def add_theme_styling(theme):
    if theme == "dark":
        # Dark mode colors
        bg_color = "#0e1117"
        text_color = "#e6edf3"
        card_bg = "#161b22"
        accent_color = "#FF4B4B" # Red for heat
        secondary_accent = "#4682B4" # Blue for work
        
        st.markdown(f"""
            <style>
                /* Dark Mode */
                .stApp {{ background-color: {bg_color}; color: {text_color}; }}
                h1, h2, h3 {{ color: {accent_color} !important; text-shadow: 1px 1px 2px rgba(0,0,0,0.5); }}
                .stButton>button {{ background: linear-gradient(90deg, {accent_color} 0%, #FF8C00 100%); color: white; }}
                .stButton>button:hover {{ transform: scale(1.05); box-shadow: 0 4px 8px rgba(0,0,0,0.5); }}
                .stDownloadButton>button {{ background: linear-gradient(90deg, {secondary_accent} 0%, #B0C4DE 100%); color: white; }}
                .stDownloadButton>button:hover {{ transform: scale(1.05); box-shadow: 0 4px 8px rgba(0,0,0,0.5); }}
                .stNumberInput, .stTextInput, .stSlider, .stSelectbox, .dataframe {{ background-color: {card_bg}; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.5); color: {text_color}; }}
                .heat-card {{ background-color: {card_bg}; border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.5); padding: 20px; margin-bottom: 20px; border-left: 5px solid {accent_color}; }}
                .stTabs [aria-selected="true"] {{ background-color: #333333; color: {accent_color}; font-weight: bold; }}
                .css-1d391kg, .css-12oz5g7 {{ background: #161b22; }} /* Sidebar background */
            </style>
        """, unsafe_allow_html=True)
    else:
        # Light mode colors (Original theme)
        st.markdown("""
        <style>
            /* Light Mode (Original Theme) */
            :root {
                --heat-red: #FF4B4B;
                --heat-orange: #FF8C00;
                --heat-light-orange: #FFDAB9;
                --work-blue: #4682B4;
                --work-light-blue: #B0C4DE;
                --text-color: #333333;
            }
            .stApp { background: linear-gradient(135deg, var(--heat-light-bg) 0%, #FFFFFF 100%); }
            h1, h2, h3 { color: var(--heat-red) !important; text-shadow: 1px 1px 2px rgba(0,0,0,0.1); }
            .stButton>button { background: linear-gradient(90deg, var(--heat-red) 0%, var(--heat-orange) 100%); color: white; }
            .stDownloadButton>button { background: linear-gradient(90deg, var(--work-blue) 0%, var(--work-light-blue) 100%); color: white; }
            .stNumberInput, .stTextInput, .stSlider, .stSelectbox, .dataframe { background-color: white; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .heat-card { background-color: white; border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); padding: 20px; margin-bottom: 20px; border-left: 5px solid var(--heat-red); }
            .stTabs [aria-selected="true"] { background-color: var(--heat-light-orange); color: var(--text-color); font-weight: bold; }
            .css-1d391kg, .css-12oz5g7 { background: linear-gradient(180deg, var(--heat-light-orange) 0%, var(--heat-orange) 100%); }
        </style>
        """, unsafe_allow_html=True)

add_theme_styling(st.session_state.theme)

# Function to create a card container
def card(title, content):
    st.markdown(f"""
    <div class="heat-card">
        <h3>{title}</h3>
        {content}
    </div>
    """, unsafe_allow_html=True)

# Function to run all three cycles for comparison
def run_all_cycles_standard(P1_kpa, T1, r, Tmax, gamma=DEFAULT_GAMMA):
    # Otto (T3=Tmax)
    res_otto = otto_cycle_standard(P1_kpa, T1, r, Tmax, gamma)
    
    # Diesel (Tmax is used to find cutoff, assuming max pressure is P3)
    # To compare fairly, we need to find the cutoff that results in the same Tmax
    # Tmax = T3_diesel. T3_diesel = T2 * rho. rho = Tmax / T2.
    T2 = T1 * r ** (gamma - 1)
    rho_diesel = Tmax / T2
    res_diesel = diesel_cycle_standard(P1_kpa, T1, r, rho_diesel, Tmax, gamma)
    
    # Dual (Tmax is T4) - using 50% CV heat fraction as a default for comparison
    res_dual = dual_cycle_standard(P1_kpa, T1, r, 1.0, 0.5, Tmax, gamma)
    
    return res_otto, res_diesel, res_dual

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="Thermodynamic Cycles Simulator", layout="wide")
st.title("🔥 Thermodynamic Cycles Simulator — Engine Models")
st.markdown("Interactive simulator with P-V, T-S, file import, and PDF export.")

# Left: inputs and file upload
with st.sidebar:
    st.header("Simulation Controls")
    
    # Date and Time Display
    now = datetime.now()
    date_str = now.strftime("%A, %B %d, %Y")
    st.markdown(f"""<div style='text-align: center; font-size: 14px; font-weight: bold; color: #333333;'>
    {date_str}
    </div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Light/Dark Mode Toggle
    st.subheader("Theme")
    col_light, col_dark = st.columns(2)
    with col_light:
        if st.button("☀️ Light Mode", use_container_width=True):
            st.session_state.theme = "light"
            st.rerun()
    with col_dark:
        if st.button("🌙 Dark Mode", use_container_width=True):
            st.session_state.theme = "dark"
            st.rerun()
            
    st.markdown("---")
    
    # Mode Selector (Renamed)
    st.subheader("Calculation Mode")
    calc_mode = st.selectbox("Select Mode", ["AIR STANDARD", "FUEL-AIR"])
    
    st.markdown("---")
    st.subheader("Cycle Parameters")

    # Common parameters for both modes
    cycle_type = st.selectbox("Cycle Type", ["Otto", "Diesel", "Dual"])
    p1_kpa = st.number_input("Initial Pressure P1 (kPa)", value=101.325)
    t1 = st.number_input("Initial Temperature T1 (K)", value=300.0)
    compression_ratio = st.number_input("Compression Ratio (r)", value=8.0, min_value=1.1)
    
    # Mode-specific inputs
    if calc_mode == "AIR STANDARD":
        gamma = st.number_input("Specific heat ratio γ", value=1.4, min_value=1.01, step=0.01)
        
        # cycle-specific
        if cycle_type == "Otto":
            t_max = st.number_input("Peak Temperature T3 (K)", value=2000.0)
            cutoff = 1.0
            cv_fraction = 1.0
        elif cycle_type == "Diesel":
            # Input is Tmax, and we calculate the cutoff ratio
            t_max = st.number_input("Peak Temperature T3 (K)", value=2000.0)
            # The cutoff ratio is calculated internally in the function
            cutoff = 1.0 # Placeholder, not used as direct input
            cv_fraction = 0.0
        else:  # Dual
            t_max = st.number_input("Peak Temperature T4 (K)", value=2000.0)
            cv_fraction = st.slider("CV heat fraction (0..1)", min_value=0.0, max_value=1.0, value=0.5)
            cutoff = 1.0 # Placeholder, not used as direct input
            
        # Set Air-Fuel parameters to None
        CV, AFR, n = None, None, None
        
    else: # FUEL-AIR
        st.info(f"Using variable specific heat: $c_v = {B1} + {K1} T$ and $c_p = {A1} + {K1} T$")
        
        # Problem-specific inputs
        CV = st.number_input("Calorific Value (CV) (kJ/kg)", value=44000.0)
        AFR = st.number_input("Air-Fuel Ratio (AFR)", value=15.0)
        n = st.number_input("Index of Compression (n)", value=1.32, min_value=1.01, step=0.01)
        
        # Set Standard parameters to None
        gamma, t_max = None, None
        
        # Cycle-specific inputs for Air-Fuel mode (only needed for Dual)
        if cycle_type == "Dual":
            cv_fraction = st.slider("CV heat fraction (0..1)", min_value=0.0, max_value=1.0, value=0.5)
            cutoff = 1.0 # Placeholder
        elif cycle_type == "Diesel":
            cutoff = 1.0 # Placeholder
            cv_fraction = 0.0
        else: # Otto
            cutoff = 1.0
            cv_fraction = 1.0
            
        
    st.markdown("---")
    st.subheader("Input from file (optional)")
    uploaded = st.file_uploader("Upload CSV or Excel with parameters", type=["csv","xlsx"])
    
    # Simplified file upload logic for the new structure (not fully implemented to avoid complexity)
    if uploaded:
        st.warning("File upload is currently only supported for the AIR STANDARD parameters.")

    st.markdown("---")
    if st.button("Run Simulation"):
        st.session_state.run_simulation = True
        st.rerun()

# ----------------------
# Main area: compute and show
# ----------------------
res = None

if st.session_state.run_simulation:

    if calc_mode == "AIR STANDARD":

        if cycle_type == "Otto":
            res = otto_cycle_standard(p1_kpa, t1, compression_ratio, t_max, gamma)

        elif cycle_type == "Diesel":
            # cutoff is calculated internally based on Tmax
            T2 = t1 * compression_ratio ** (gamma - 1)
            rho_diesel = t_max / T2
            res = diesel_cycle_standard(p1_kpa, t1, compression_ratio, rho_diesel, t_max, gamma)

        elif cycle_type == "Dual":
            # cutoff is calculated internally based on Tmax and cv_fraction
            res = dual_cycle_standard(p1_kpa, t1, compression_ratio, 1.0, cv_fraction, t_max, gamma)

    elif calc_mode == "FUEL-AIR":

        # FUEL-AIR mode calculation
        res = air_fuel_base_calc(p1_kpa, t1, compression_ratio, CV, AFR, n, cycle_type, cutoff, cv_fraction)

        # Comparison calculation
        Cv_const = B1  # Use the constant part of Cv
        P_max_const_cv = air_fuel_const_cv_comparison(
            p1_kpa, t1, compression_ratio, CV, AFR, n, Cv_const, cycle_type, cutoff, cv_fraction
        )
        res["P_max_const_cv"] = P_max_const_cv

if res is None:
    st.stop()


# ----------------------
tab_results, tab_info, tab_formulas = st.tabs(["📊 Results & Diagrams", "💡 Cycle Information", "📜 Formulas & Theory"])

with tab_results:
    st.subheader("Summary Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cycle", res["name"])
    col2.metric("Net Work (kJ/kg)", f"{res['W_net']:.0f}")
    col3.metric("Heat In (kJ/kg)", f"{res['Q_in']:.0f}")
    col4.metric("Efficiency", f"{res['eta_analytic']*100:.0f} %")
    
    # Add MEP to Summary Metrics
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("MEP (kPa)", f"{res['MEP']:.2f}")

    if calc_mode == "FUEL-AIR":
        st.subheader("FUEL-AIR Results Summary")
        
        # Display key temperatures and pressures
        col_t2, col_t3, col_t4 = st.columns(3)
        col_t2.metric("T2 (End of Comp)", f"{res['T2']:.0f} K")
        col_t3.metric("T3 (End of CV Heat)", f"{res['T3']:.0f} K")
        col_t4.metric("T4 (End of CP Heat)", f"{res['T4']:.0f} K")
        
        col_p2, col_p3, col_p4 = st.columns(3)
        col_p2.metric("P2 (End of Comp)", f"{res['P2']:.0f} kPa")
        col_p3.metric("P3 (End of CV Heat)", f"{res['P3']:.0f} kPa")
        col_p4.metric("P4 (End of CP Heat)", f"{res['P4']:.0f} kPa")
        
        st.markdown("---")
        st.metric("Maximum Pressure (Variable $c_v$)", f"{max(res['P3'], res['P4']):.0f} kPa")
        st.metric("Maximum Pressure (Constant $c_v={B1}$)", f"{res['P_max_const_cv']:.0f} kPa")
        
    st.subheader("State Points")
    # Removed Entropy from the display table
    states_df = pd.DataFrame([
        {"State": k, "Pressure (kPa)": v["P"], "Volume (norm)": v["V"], "Temperature (K)": v["T"]}
        for k, v in res["states"].items()
    ])
    # Rounding the DataFrame values to the nearest integer
    states_df["Pressure (kPa)"] = states_df["Pressure (kPa)"].round(0)
    states_df["Volume (norm)"] = states_df["Volume (norm)"].round(3) # Keep some precision for volume
    states_df["Temperature (K)"] = states_df["Temperature (K)"].round(0)
    st.dataframe(states_df.set_index("State"))

    st.subheader("Diagrams")
    pv_col, ts_col = st.columns(2)

    # Plots: PV, TS, Step processes
    fig_pv = plot_pv(res["states"], title=f"{res['name']} - P-V", gamma=res.get("gamma", DEFAULT_GAMMA))
    fig_ts = plot_ts(res["states"], res["s"], title=f"{res['name']} - T-S (approx)", gamma=res.get("gamma", DEFAULT_GAMMA))
    fig_steps = plot_step_processes(res["states"], res.get("gamma", DEFAULT_GAMMA), cycle_name=res["name"])
    
    with pv_col:
        st.caption("Pressure-Volume diagram")
        st.pyplot(fig_pv)

    with ts_col:
        st.caption("Temperature-Entropy diagram (approx)")
        st.pyplot(fig_ts)

    st.caption("Step-by-step process plots")
    st.pyplot(fig_steps)
    
    # Removed HRR plot display

with tab_info:
    st.markdown("## 💡 Cycle Information and Comparison")
    
    # New section for "sweet information" - General thermodynamic principles
    card("Thermodynamic Principles (Sweet Information)", """
    The **Air Standard** cycle assumes air is the working fluid, behaves as an ideal gas, and has constant specific heats. This provides a simple, theoretical upper limit for efficiency.
    
    The **Fuel-Air** cycle (or Variable Specific Heat cycle) provides a more realistic model by accounting for the change in specific heats with temperature, which is crucial at the high temperatures encountered in real engines.
    
    **Key Relationships (AIR STANDARD):**
    *   **Efficiency ($\eta$)**: is primarily governed by the compression ratio ($r$). Higher $r$ leads to higher efficiency.
    *   **Maximum Pressure ($P_{\text{max}}$)**: is a critical design constraint. For a given $r$ and heat input, the Otto cycle (constant volume heat addition) achieves the highest peak pressure.
    *   **Net Work ($W_{\text{net}}$)**: is the area enclosed by the cycle on the P-V diagram.
    """)
    
    if calc_mode == "FUEL-AIR":
        card("Key Relationships (FUEL-AIR)", """
        *   **Specific Heat Variation**: The specific heats ($c_p$ and $c_v$) increase with temperature, which is accounted for in the calculations. This leads to a **lower** actual efficiency compared to the theoretical AIR STANDARD cycle.
        *   **Polytropic Process**: The compression and expansion processes are modeled as polytropic ($PV^n = \text{const}$) with an index $n$ (instead of $\gamma$) to account for heat losses and friction in real engines.
        *   **Note**: This model does not account for Dissociation, which occurs at very high temperatures (above $\approx 2000 \text{ K}$) and further reduces maximum temperature and efficiency.
        """)
    
    # Existing cycle descriptions
    if cycle_type == "Otto":
        card("Otto Cycle (Constant Volume Heat Addition)", """
        The **Otto cycle** is the thermodynamic cycle that describes the functioning of a typical spark-ignition piston engine.
        It consists of four processes:
        1. **Isentropic compression** (1→2): The air-fuel mixture is compressed.
        2. **Constant volume heat addition** (2→3): Combustion occurs, increasing pressure and temperature.
        3. **Isentropic expansion** (3→4): The power stroke.
        4. **Constant volume heat rejection** (4→1): Exhaust valve opens, rejecting heat.
        """)
        
    elif cycle_type == "Diesel":
        card("Diesel Cycle (Constant Pressure Heat Addition)", """
        The **Diesel cycle** is the thermodynamic cycle that describes the functioning of a compression-ignition piston engine.
        It consists of four processes:
        1. **Isentropic compression** (1→2).
        2. **Constant pressure heat addition** (2→3): Fuel is injected and burns, maintaining constant pressure.
        3. **Isentropic expansion** (3→4): The power stroke.
        4. **Constant volume heat rejection** (4→1).
        """)
        
    else: # Dual
        card("Dual Cycle (Combined Heat Addition)", """
        The **Dual cycle** is a combination of the Otto and Diesel cycles, often used as a more accurate model for modern high-speed compression-ignition engines.
        Heat addition occurs in two stages:
        1. **Constant volume heat addition** (2→3).
        2. **Constant pressure heat addition** (3→4).
        The rest of the processes are: Isentropic compression (1→2) and Isentropic expansion (4→5).
        """)
        
    # Comparison Table (Only for AIR STANDARD mode for simplicity and fair comparison)
    if calc_mode == "AIR STANDARD":
        st.subheader("Cycle Efficiency Comparison (AIR STANDARD)")
        st.markdown("Comparison of Otto, Diesel, and Dual cycles for the same compression ratio ($r$) and maximum temperature ($T_{max}$):")
        
        try:
            res_otto, res_diesel, res_dual = run_all_cycles_standard(p1_kpa, t1, compression_ratio, t_max, gamma)
            
            comparison_data = {
                "Cycle": [res_otto["name"], res_diesel["name"], res_dual["name"]],
                "Efficiency ($\eta$)": [f"{res_otto['eta_analytic']*100:.0f} %", f"{res_diesel['eta_analytic']*100:.0f} %", f"{res_dual['eta_analytic']*100:.0f} %"],
                "Net Work ($W_{net}$)": [f"{res_otto['W_net']:.0f}", f"{res_diesel['W_net']:.0f}", f"{res_dual['W_net']:.0f}"],
                "Max Pressure ($P_{max}$)": [f"{max(res_otto['states'][3]['P'], res_otto['states'][4]['P']):.0f}", f"{max(res_diesel['states'][3]['P'], res_diesel['states'][4]['P']):.0f}", f"{max(res_dual['states'][3]['P'], res_dual['states'][4]['P']):.0f}"],
            }
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df.set_index("Cycle"))
            
            st.markdown("""
            **Key Observation:** For the same compression ratio and maximum temperature, the theoretical efficiency order is:
            $$\eta_{\\text{Otto}} > \eta_{\\text{Dual}} > \eta_{\\text{Diesel}}$$
            """)
            
        except Exception as e:
            st.error(f"Could not generate comparison table: {e}")


with tab_formulas:
    st.header("📜 Summary of Formulas and Theory")
    
    st.subheader("1. AIR STANDARD Cycle Efficiency (Constant Specific Heat)")
    
    card("Otto Cycle Efficiency", r"""
    $$\eta_{\text{Otto}} = 1 - \frac{1}{r^{\gamma-1}}$$
    Where $r$ is the compression ratio and $\gamma$ is the specific heat ratio.
    """)
    
    card("Diesel Cycle Efficiency", r"""
    $$\eta_{\text{Diesel}} = 1 - \frac{1}{r^{\gamma-1}} \left[ \frac{\rho^\gamma - 1}{\gamma(\rho - 1)} \right]$$
    Where $\rho$ is the cutoff ratio ($V_3/V_2$).
    """)
    
    card("Dual Cycle Efficiency", r"""
    $$\eta_{\text{Dual}} = 1 - \frac{1}{r^{\gamma-1}} \left[ \frac{\alpha \rho^\gamma - 1}{(\alpha - 1) + \alpha \gamma (\rho - 1)} \right]$$
    Where $\alpha$ is the pressure ratio ($P_3/P_2$) and $\rho$ is the cutoff ratio ($V_4/V_3$).
    """)

    st.subheader("2. FUEL-AIR Cycle (Variable Specific Heat)")

    # New section for MEP
    st.subheader("3. Mean Effective Pressure (MEP)")
    
    card("Mean Effective Pressure (MEP)", r"""
    Mean Effective Pressure (MEP) is a hypothetical constant pressure that, if applied to the piston during the power stroke,
    would produce the same net work ($W_{\text{net}}$) as the actual thermodynamic cycle.
    
    MEP relates the net work output to the swept volume ($V_{\text{swept}}$), making it independent of engine size.
    
    **Formula:**
    $$\text{MEP} = \frac{W_{\text{net}}}{V_{\text{swept}}}$$
    
    In the Air Standard Cycle:  
    $$V_{\text{swept}} = V_1 - V_2$$
    
    **P-V Representation:**
    MEP appears as the height of a rectangle whose area equals net work.
    """)

    card("Variable Specific Heat Functions (Updated)", r"""
    The model uses variable specific heat functions (in $\text{kJ/kg.K}$),
    based on the form:
    $$c_v(T) = {B1} + {K1} T$$
    $$c_p(T) = {A1} + {K1} T$$
    
    Gas constant:
    $$R_{\text{air}} = c_p - c_v = a_1 - b_1 = {R_AIR}$$
    
    **Internal Energy and Enthalpy:**
    $$u(T) = b_1 T + \frac{k_1}{2} T^2$$
    $$h(T) = a_1 T + \frac{k_1}{2} T^2$$
    """.replace("{B1}", str(B1))
       .replace("{K1}", str(K1))
       .replace("{A1}", str(A1))
       .replace("{R_AIR}", str(R_AIR)))

    card("Heat Addition Calculation", r"""
    Heat added per kg of mixture:
    $$Q_{\text{in}} = \frac{\text{Calorific Value}}{\text{AFR} + 1}$$
    
    - **Constant Volume Heat ($Q_{\text{cv}}$)**:
      $$Q_{\text{cv}} = \Delta u = u(T_{\text{end}}) - u(T_{\text{start}})$$
    
    - **Constant Pressure Heat ($Q_{\text{cp}}$)**:
      $$Q_{\text{cp}} = \Delta h = h(T_{\text{end}}) - h(T_{\text{start}})$$
    """)


# ----------------------
# Export PDF report (Updated to remove HRR and Entropy)
# ----------------------
def build_pdf_bytes(res, fig_pv, fig_ts, fig_steps):
    buf = BytesIO()
    try:
        with PdfPages(buf) as pdf:
            # Page 1: summary text + table
            fig_text = plt.figure(figsize=(8.27, 11.69))  # A4
            fig_text.clf()
            txt = fig_text.text(0.1, 0.95, f"Thermodynamic Cycle Report: {res['name']}", fontsize=16, weight='bold')
            # summary lines
            lines = [
                f"Cycle: {res['name']}",
                f"Net Work (approx): {res['W_net']:.6f} (kJ/kg unit)",
                f"Heat In (approx): {res['Q_in']:.6f} (kJ/kg)",
                f"Efficiency (approx): {res['eta_analytic']*100:.2f} %",
                "",
                "State points (P kPa, V normalized, T K):"
            ]
            y = 0.88
            for L in lines:
                fig_text.text(0.05, y, L, fontsize=10)
                y -= 0.03
            # add table chunk (without Entropy)
            tbl_df = pd.DataFrame([
                {"State": k, "P (kPa)": v["P"], "V": v["V"], "T (K)": v["T"]}
                for k,v in res["states"].items()
            ])
            # render table as image in figure
            ax = fig_text.add_axes([0.05, 0.1, 0.9, 0.5])
            ax.axis('off')
            tbl = ax.table(cellText=tbl_df.values, colLabels=tbl_df.columns, loc='center')
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            tbl.scale(1, 1.2)
            pdf.savefig(fig_text)
            plt.close(fig_text)

            # Page 2: PV
            pdf.savefig(fig_pv)
            plt.close(fig_pv)

            # Page 3: TS
            pdf.savefig(fig_ts)
            plt.close(fig_ts)

            # Page 4: step processes
            pdf.savefig(fig_steps)
            plt.close(fig_steps)
            
        buf.seek(0)
        return buf.read()
    except Exception as e:
        st.error(f"Error in PDF generation: {e}")
        return None

with tab_formulas:
    st.subheader("Export Report (PDF)")
    
    if st.button("Generate & Download PDF Report"):
        try:
            # Re-run plots to ensure they are available for PDF generation
            gamma_val = res.get("gamma", DEFAULT_GAMMA)
            fig_pv = plot_pv(res["states"], title=f"{res['name']} - P-V", gamma=gamma_val)
            fig_ts = plot_ts(res["states"], res["s"], title=f"{res['name']} - T-S (approx)", gamma=gamma_val)
            fig_steps = plot_step_processes(res["states"], gamma_val, cycle_name=res["name"])
            
            # Updated PDF function call
            pdf_bytes = build_pdf_bytes(res, fig_pv, fig_ts, fig_steps)
            if pdf_bytes:
                st.download_button("Download Report (PDF)", data=pdf_bytes, file_name=f"{res['name']}_report.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"Failed to create PDF: {e}")

st.markdown("---")
st.caption("Notes: All thermodynamic calculations are approximate and for educational purposes. Units use kPa, K, and kJ per normalized unit volume/mass approximation.")
