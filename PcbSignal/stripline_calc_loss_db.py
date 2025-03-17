import numpy as np

# Constants
sigma = 5.96e7  # Copper conductivity (S/m)
mu_0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)
Z_0 = 50  # Characteristic impedance (ohms)
W = 6e-3  # Trace width in inches (6 mil)
t = 0.7e-3  # Copper thickness in inches (0.7 mil)
b = 16e-3  # Dielectric height in inches (16 mil)

# Frequencies in GHz
frequencies_GHz = np.array([0.01, 0.1, 1, 8, 16, 25, 32, 53])
frequencies_Hz = frequencies_GHz * 1e9  # Convert to Hz

# Material data from table
materials = {
    "FR-4": {"roughness": 1e-6, "Dk": [4.5, 4.5, 4.4, 4.4, 4.4, 4.3, 4.3], "Df": [0.02, 0.02, 0.022, 0.023, 0.023, 0.025, 0.025], "freqs": [1, 10, 16, 25, 28, 53, 56]},
    "Rogers RO4003C": {"roughness": 0.4e-6, "Dk": [3.38, 3.38, 3.35, 3.33, 3.32, 3.3, 3.3], "Df": [0.0025, 0.0026, 0.0027, 0.0029, 0.003, 0.0035, 0.0036], "freqs": [1, 10, 16, 25, 28, 53, 56]},
    "Rogers RO4350B": {"roughness": 0.4e-6, "Dk": [3.48, 3.45, 3.42, 3.4, 3.38, 3.35, 3.34], "Df": [0.003, 0.0032, 0.0034, 0.0035, 0.0037, 0.004, 0.0041], "freqs": [1, 10, 16, 25, 28, 53, 56]},
    "Isola I-Tera MT40": {"roughness": 0.4e-6, "Dk": [3.35, 3.34, 3.32, 3.3, 3.28, 3.25, 3.24], "Df": [0.003, 0.0031, 0.0033, 0.0034, 0.0036, 0.004, 0.0041], "freqs": [1, 10, 16, 25, 28, 53, 56]},
    "Nelco N4000-13": {"roughness": 0.5e-6, "Dk": [3.4, 3.39, 3.37, 3.35, 3.33, 3.3, 3.3], "Df": [0.003, 0.0032, 0.0034, 0.0035, 0.0037, 0.0039, 0.004], "freqs": [1, 10, 16, 25, 28, 53, 56]},
    "Teflon (PTFE)": {"roughness": 0.2e-6, "Dk": [2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1], "Df": [0.0002, 0.0003, 0.0004, 0.0004, 0.0005, 0.0006, 0.0007], "freqs": [1, 10, 16, 25, 28, 53, 56]},
    "Megtron 6": {"roughness": 0.3e-6, "Dk": [3.5, 3.48, 3.46, 3.45, 3.43, 3.4, 3.4], "Df": [0.0025, 0.0026, 0.0028, 0.0029, 0.003, 0.0033, 0.0034], "freqs": [1, 10, 16, 25, 28, 53, 56]},
}

# Conductor Loss Base Calculation (independent of material)
omega = 2 * np.pi * frequencies_Hz
skin_depth = np.sqrt(2 / (omega * mu_0 * sigma))  # in meters
Rs = np.sqrt(omega * mu_0 / (2 * sigma))  # ohms/square
alpha_c_smooth = (Rs / (Z_0 * W)) * 8.686  # dB/inch (smooth copper)

# Function to interpolate Dk and Df
def interpolate_values(freq, freq_points, values):
    return np.interp(freq, freq_points, values)

# Calculate and print table
print("Material              | Freq (GHz) | Dk      | Df      | Dielectric Loss | Conductor Loss (Smooth) | Conductor Loss (Rough) | Total Loss")
print("-" * 110)

for material, data in materials.items():
    roughness = data["roughness"]
    freq_points = data["freqs"]
    Dk_values = data["Dk"]
    Df_values = data["Df"]
    
    for i, f in enumerate(frequencies_GHz):
        # Interpolate Dk and Df for the current frequency
        Dk = interpolate_values(f, freq_points, Dk_values)
        Df = interpolate_values(f, freq_points, Df_values)
        
        # Dielectric Loss
        alpha_d = 0.911 * np.sqrt(Dk) * Df * f
        
        # Conductor Loss with Roughness
        K = 1 + (2 / np.pi) * np.arctan(1.4 * (roughness / skin_depth[i])**2)
        alpha_c_rough = alpha_c_smooth[i] * K
        
        # Total Loss
        alpha_total = alpha_d + alpha_c_rough
        
        # Print row
        print(f"{material:<20} | {f:>9.2f} | {Dk:>6.2f} | {Df:>6.4f} | {alpha_d:>14.4f} | {alpha_c_smooth[i]:>19.4f} | {alpha_c_rough:>18.4f} | {alpha_total:>10.4f}")
    print("-" * 110)