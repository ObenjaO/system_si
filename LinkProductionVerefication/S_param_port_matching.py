import numpy as np
import skrf as rf
import matplotlib.pyplot as plt
import argparse
import os
import sys
from s4p_self_cascading import to_db, mixed_mode_s_params

# Convert Z_old and Z_new to numbers (handle strings, bools, etc.)
def to_complex(z):
    if isinstance(z, (int, float, complex)):
        return complex(z)
    elif isinstance(z, str):
        if z.lower() in ('true', 'false'):
            raise ValueError(f"Impedance cannot be a boolean string: {z}")
        return complex(z.replace('j', '')) if 'j' in z else complex(float(z))
    elif isinstance(z, bool):
        raise ValueError(f"Impedance cannot be a boolean: {z}")
    else:
        raise TypeError(f"Invalid impedance type: {type(z)}")
    
def renormalize_s_matrix(S, Z_old, Z_new):
    n_freqs, n_ports, _ = S.shape
    S_adjusted = np.zeros_like(S, dtype=complex)
    
    Z_old = to_complex(Z_old)
    Z_new = to_complex(Z_new)

    # Handle scalar impedance (same for all ports)
    if np.isscalar(Z_old):
        Z_old = [Z_old] * n_ports
    if np.isscalar(Z_new):
        Z_new = [Z_new] * n_ports
    
    Gamma = [(zn - zo) / (zn + zo) for zn, zo in zip(Z_new, Z_old)]
    Gamma_diag = np.diag(Gamma)
    I = np.eye(n_ports)
    A = I - Gamma_diag
    B = I + Gamma_diag
    B_inv = np.linalg.inv(B)
    
    for i in range(n_freqs):
        S_adjusted[i] = A @ B_inv @ S[i] @ B_inv @ A
    
    return S_adjusted

def renormalize_s_matrix_alt(S, Z_old, Z_new):
    n_freqs, n_ports, _ = S.shape
    S_adjusted = np.zeros_like(S, dtype=complex)
    #Z_old = [Z_old] * n_ports if np.isscalar(Z_old) else Z_old
    #Z_new = [Z_new] * n_ports if np.isscalar(Z_new) else Z_new
    Z_old = to_complex(Z_old)
    Z_new = to_complex(Z_new)

    # Handle scalar impedance (same for all ports)
    if np.isscalar(Z_old):
        Z_old = [Z_old] * n_ports
    if np.isscalar(Z_new):
        Z_new = [Z_new] * n_ports

    for i in range(n_freqs):
        # Convert S to Z-matrix
        Z0 = np.diag([Z_old[j] for j in range(n_ports)])
        I = np.eye(n_ports)
        Z = Z0 @ (I + S[i]) @ np.linalg.inv(I - S[i])
        # Adjust reference impedance
        Z_new_diag = np.diag([Z_new[j] for j in range(n_ports)])
        S_adjusted[i] = (Z - Z_new_diag) @ np.linalg.inv(Z + Z_new_diag)
    
    return S_adjusted

def main():
    res = True
    PASSIVITY_TOL = 1e-6

    parser = argparse.ArgumentParser(description="Change income S parameter to new port impedance.")
    
    # Adding arguments for file path and debug mode
    parser.add_argument('file', type=str, help="The full path to the file to process.")
    parser.add_argument('New_impedance', type=float, help="The new impedance of the port.")
    parser.add_argument('debug', type=bool, help="Enable or disable debug mode (True/False).")
    
    # Parsing the command-line arguments
    args = parser.parse_args()
    if len(vars(args)) != 3:
        print("Error: Expected 2 arguments (file path and debug mode).")
        return
    # Check that the first argument is a string (it is because we specified 'type=str')
    file_path = args.file
    if not isinstance(file_path, str):
        print("Error: The file path should be a string.")
        return
    
    file = sys.argv[1]
    debug_mode = sys.argv[3]
    # Load the original S-parameter file (e.g., 4-port)
    # Extracting the file name
    file_name = os.path.basename(args.file)
    ntwk = rf.Network(args.file)
    print(f"Original Z0: {ntwk.z0[0, :]} ohms want to change to ")  # Should be [50, 50, 50, 50]

    # Define impedances (all ports to 45 ohm)
    Z_old = ntwk.z0[0, 0]  # Scalar, applies to all ports
    Z_new = sys.argv[2]  # Scalar, applies to all ports
        
    # Check that the second argument is either 'True' or 'False'
    debug_mode = debug_mode.strip()
    if debug_mode not in ["True", "False"]:
        print("Error: The debug mode should be either 'True' or 'False'. Received: {repr(debug_mode)}")
        print(debug_mode)
        return
    
    # Renormalize
    #S_adjusted = renormalize_s_matrix(ntwk.s, Z_old, Z_new)
    S_adjusted = renormalize_s_matrix_alt(ntwk.s, Z_old, Z_new)
    # Normalize S21 to match original magnitude at a reference frequency (e.g., DC)
    #scale_factor = np.abs(ntwk.s[0, 1, 0]) / np.abs(S_adjusted[0, 1, 0])
    #S_adjusted *= scale_factor
    ntwk_renorm = rf.Network(frequency=ntwk.frequency, s=S_adjusted, z0=Z_new)
    print(f"Adjusted Z0: {ntwk_renorm.z0[0, :]} ohms")  # Should be [45, 45, 45, 45]
    if debug_mode == "True":
        Freq = ntwk.f
        if ntwk.nports == 4:
            mm_ntwk = mixed_mode_s_params(ntwk.s)
            mm_ntwk_renorm = mixed_mode_s_params(ntwk_renorm.s)
            IL1 = to_db(mm_ntwk['sdd21'])
            IL2 = to_db(mm_ntwk['sdd12'])
            RL1 = to_db(mm_ntwk['sdd11'])
            RL2 = to_db(mm_ntwk['sdd22'])
            IL1_adj = to_db(mm_ntwk_renorm['sdd21'])
            IL2_adj = to_db(mm_ntwk_renorm['sdd12'])
            RL1_adj = to_db(mm_ntwk_renorm['sdd11'])
            RL2_adj = to_db(mm_ntwk_renorm['sdd22'])
        else:
            IL1 = to_db(ntwk.s[:, 1, 0])
            IL2 = to_db(ntwk.s[:, 0, 1])
            RL1 = to_db(ntwk.s[:, 0, 0])
            RL2 = to_db(ntwk.s[:, 1, 1])
            IL1_adj = to_db(ntwk_renorm.s[:, 1, 0])
            IL2_adj = to_db(ntwk_renorm.s[:, 0, 1])
            RL1_adj = to_db(ntwk_renorm.s[:, 0, 0])
            RL2_adj = to_db(ntwk_renorm.s[:, 1, 1])
    # Plot S21 (example for ports 1->2, adjust indices as needed)
    freqs = ntwk.f
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    
        
        
    axes[0].plot(Freq, IL1, label=f"S21 ({Z_old} ohm)-ref.")
    axes[0].plot(Freq, IL1_adj, label=f"S21 ({Z_new} ohm)-matched")
    axes[0].set_xlabel("Frequency (Hz)", fontsize=16)
    axes[0].set_ylabel("Magnitude (dB)", fontsize=16)
    axes[0].set_title("S21: compare port matching vs Reference")
    axes[0].legend()
    axes[0].grid(True)
    plt.legend()
    plt.grid()
    
    axes[1].plot(Freq, RL1, label=f"S11 ({Z_old} ohm)-ref.")
    axes[1].plot(Freq, RL1_adj, label=f"S11 ({Z_new} ohm)-matched")
    axes[1].set_xlabel("Frequency (Hz)", fontsize=16)
    axes[1].set_ylabel("Magnitude (dB)", fontsize=16)
    axes[1].set_title("S11: compare port matching vs Reference")

    axes[1].grid(True)
    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()
    # Save the adjusted S-parameters
    ntwk_renorm.write_touchstone(f"{file_name}_adjusted_{Z_new}ohm.s4p")
    
    

if __name__ == "__main__":
    main()