from jax import config
from jax import lax
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.experimental.ode import odeint

pi = jnp.pi

def dma_mr_uncertain_pressures(input_vec, output_vec):
    
    
    Pt = input_vec[0]
    Ps = input_vec[1]

    T = 1173.15
    selec = 1500.00
    benz = output_vec[0]
    conv = output_vec[1]

    
    # Kinetic  and general parameters
    R = 8.314e6                 # [Pa.cm³/(K.mol.)]
    k1 = 0.04                   # [s-¹]
    k1_Inv = 6.40e6             # [cm³/s-mol]
    k2 = 4.20                   # [s-¹]
    k2_Inv = 56.38              # [cm³/s-mol]

    # Molecular weights
    MM_B = 78.00                # [g/mol] Benzene


    # Fixed Reactor Values
    Q = 3600 * 0.01e-4          # [mol/(h.cm².atm1/4)]

    # Tube side
    # Pt = 101325.0               # Pressure [Pa](1atm)
    v0 = 3600 * (2 / 15)        # Vol. Flowrate [cm³ h-¹]
    Ft0 = Pt * v0 / (R * T)     # Initial molar flowrate[mol/h] - Pure CH4

    # Shell side
    # Ps = 101325.0               # Pressure [Pa](1atm)
    ds = 3                      # Diameter[cm]
    v_He = 3600 * (1 / 6)       # Vol. flowrate[cm³/h]
    F_He = Ps * v_He / (R * T)  # Sweep gas molar flowrate [mol/h]
    
    

    # Initial conditions
    y0 = jnp.hstack((Ft0, jnp.zeros(7)))
    rtol, atol = 1e-10, 1e-12

    z = jnp.linspace(0, 17.2283, 2000)
    # z = jnp.linspace(0, 30, 2000)
    F = odeint(dma_mr_ST, y0, z, selec,F_He, Ft0, v0, Pt, Ps, k1,k1_Inv,
               k2,k2_Inv, Q, rtol=rtol, atol=atol)

    
    F_C6H6 = ((F[-1, 3] * 1000) * MM_B)
    X_CH4  = (100 * (Ft0 - F[-1, 0] - F[-1, 4]) / Ft0)
    
    LHS1 = benz - F_C6H6
    LHS2 = conv - X_CH4

    return jnp.array([LHS1, LHS2])


def dma_mr_uncertain_flows(input_vec, output_vec):
    

    T = 1173.15
    selec = 1500.00
    benz = output_vec[0]
    conv = output_vec[1]
    v0 = input_vec[0] 
    v_He = input_vec[1] 
    
    # Kinetic  and general parameters
    R = 8.314e6                 # [Pa.cm³/(K.mol.)]
    k1 = 0.04                   # [s-¹]
    k1_Inv = 6.40e6             # [cm³/s-mol]
    k2 = 4.20                   # [s-¹]
    k2_Inv = 56.38              # [cm³/s-mol]

    # Molecular weights
    MM_B = 78.00                # [g/mol] Benzene


    # Fixed Reactor Values
    Q = 3600 * 0.01e-4          # [mol/(h.cm².atm1/4)]

    # Tube side
    Pt = 101325.0               # Pressure [Pa](1atm)
    # v0 = 3600 * (2 / 15)        # Vol. Flowrate [cm³ h-¹]
    Ft0 = Pt * v0 / (R * T)     # Initial molar flowrate[mol/h] - Pure CH4

    # Shell side
    Ps = 101325.0               # Pressure [Pa](1atm)
    ds = 3                      # Diameter[cm]
    # v_He = 3600 * (1 / 6)       # Vol. flowrate[cm³/h]
    F_He = Ps * v_He / (R * T)  # Sweep gas molar flowrate [mol/h]
    
    

    # Initial conditions
    y0 = jnp.hstack((Ft0, jnp.zeros(7)))
    rtol, atol = 1e-10, 1e-10

    z = jnp.linspace(0, 17.2283, 2000)
    # z = jnp.linspace(0, 30, 2000)
    F = odeint(dma_mr_ST, y0, z, selec,F_He, Ft0, v0, Pt, Ps, k1,k1_Inv,
               k2,k2_Inv, Q, rtol=rtol, atol=atol)

    
    F_C6H6 = ((F[-1, 3] * 1000) * MM_B)
    X_CH4  = (100 * (Ft0 - F[-1, 0] - F[-1, 4]) / Ft0)
    
    LHS1 = benz - F_C6H6
    LHS2 = conv - X_CH4

    return jnp.array([LHS1, LHS2])



def dma_mr_uncertain_flows_check(u):
    

    T = 1173.15
    selec = 1500.00

    v0 = u[0] 
    v_He = u[1] 
    
    # Kinetic  and general parameters
    R = 8.314e6                 # [Pa.cm³/(K.mol.)]
    k1 = 0.04                   # [s-¹]
    k1_Inv = 6.40e6             # [cm³/s-mol]
    k2 = 4.20                   # [s-¹]
    k2_Inv = 56.38              # [cm³/s-mol]
    


    # Molecular weights
    MM_B = 78.00                # [g/mol] Benzene


    # Fixed Reactor Values
    Q = 3600 * 0.01e-4          # [mol/(h.cm².atm1/4)]

    # Tube side
    Pt = 101325.0               # Pressure [Pa](1atm)
    # v0 = 3600 * (2 / 15)        # Vol. Flowrate [cm³ h-¹]
    Ft0 = Pt * v0 / (R * T)     # Initial molar flowrate[mol/h] - Pure CH4

    # Shell side
    Ps = 101325.0               # Pressure [Pa](1atm)
    ds = 3                      # Diameter[cm]
    # v_He = 3600 * (1 / 6)       # Vol. flowrate[cm³/h]
    F_He = Ps * v_He / (R * T)  # Sweep gas molar flowrate [mol/h]
    
    

    # Initial conditions
    y0 = jnp.hstack((Ft0, jnp.zeros(7)))
    rtol, atol = 1e-10, 1e-10

    z = jnp.linspace(0, 17.2283, 2000)
    # z = jnp.linspace(0, 30, 2000)
    F = odeint(dma_mr_ST, y0, z, selec,F_He, Ft0, v0, Pt, Ps, k1,k1_Inv,
               k2,k2_Inv, Q, rtol=rtol, atol=atol)

    
    F_C6H6 = ((F[-1, 3] * 1000) * MM_B)
    X_CH4  = (100 * (Ft0 - F[-1, 0] - F[-1, 4]) / Ft0)
    
    LHS1 = F_C6H6
    LHS2 = X_CH4

    return jnp.array([LHS1, LHS2])



def dma_mr_ST(F, z, selec, F_He, Ft0, v0, Pt, Ps, k1,k1_Inv,k2,k2_Inv, Q):
    

    dt = 0.5634
    # dt = 1
    At = 0.25 * jnp.pi * (dt ** 2)  # Cross sectional area[cm³]
    
    # Avoid negative flows that can happen in the first integration steps.
    # Consequently this avoids that any molar balance (^ 1/4 terms) generates
    # complex numbers.
    F = jnp.where(F <= 1e-9, 1e-9, F)
    

    
    # Evaluate total flowrate in tube & shell.
    Ft = F[0:4].sum()
    Fs = F[4:].sum() + F_He
    v = v0 * (Ft / Ft0)

    # Concentrations from molar flowrates [mol/cm3]
    C = F[:4] / v
    # Partial pressures - Tube & Shell [mol/cm3]

    P0t = (Pt / 101325) * (F[0] / Ft)
    P1t = (Pt / 101325) * (F[1] / Ft)
    P2t = (Pt / 101325) * (F[2] / Ft)
    P3t = (Pt / 101325) * (F[3] / Ft)

    P0s = (Ps / 101325) * (F[4] / Fs)
    P1s = (Ps / 101325) * (F[5] / Fs)
    P2s = (Ps / 101325) * (F[6] / Fs)
    P3s = (Ps / 101325) * (F[7] / Fs)
    
    

    
    r0 = 3600 * k1 * C[0] * (1 - ((k1_Inv * C[1] * C[2] ** 2) / 
                                  (k1 * (C[0])**2 )))
    

    # This replicates an if statement whenever the concentrations are near zero
    C0_aux = C[0]
    r0 = jnp.where(C0_aux <= 1e-9, 0, r0)
    
    r1 = 3600 * k2 * C[1] * (1 - ((k2_Inv * C[3] * C[2] ** 3) / 
                                  (k2 * (C[1])**3 )))
    

    # Same as before
    C1_aux = C[1]
    r1 = jnp.where(C1_aux <= 1e-9 , 0, r1)  

    # Molar balances adjustment with experimental data.
    eff = 0.9
    vb = 0.5
    Cat = (1 - vb) * eff

    # Molar balances dFdz - Tube (0 to 3) & Shell (4 to 7)
    dF0 = -Cat * r0 * At - (Q / selec) * ((P0t ** 0.25) - (P0s ** 0.25)) * pi * dt
    dF1 = 1 / 2 * Cat * r0 * At - Cat * r1 * At - (Q / selec) * ((P1t ** 0.25) - (P1s ** 0.25)) * pi * dt
    dF2 = Cat * r0 * At + Cat * r1 * At- (Q) * ((P2t ** 0.25) - (P2s ** 0.25)) * pi * dt
    dF3 = (1 / 3) * Cat * r1 * At - (Q / selec) * ((P3t ** 0.25) - (P3s ** 0.25)) * pi * dt
    dF4 = (Q / selec) * ((P0t ** 0.25) - (P0s ** 0.25)) * pi * dt
    dF5 = (Q / selec) * ((P1t ** 0.25) - (P1s ** 0.25)) * pi * dt
    dF6 = (Q) * ((P2t ** 0.25) - (P2s ** 0.25)) * pi * dt
    dF7 = (Q / selec) * ((P3t ** 0.25) - (P3s ** 0.25)) * pi * dt
    
    dFdz = jnp.array([ dF0, dF1, dF2, dF3, dF4, dF5, dF6, dF7 ])

    return dFdz