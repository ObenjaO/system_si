import numpy as np
import matplotlib.pyplot as plt

# Constants
Dk = 4.2#3.7  # Dielectric constant (Megtron 6)
Df = 0.02#0.002  # Dissipation factor (Megtron 6)
sigma = 5.96e7  # Copper conductivity (S/m)
mu_0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)
Z_0 = 50  # Characteristic impedance (ohms)
W = 5e-3  # Trace width in inches (5 mil)
roughness = 1.5e-6  # Surface roughness RMS in meters (1.5 um)
c = 1.18e10  # Speed of light in inches/s

# Frequencies in GHz
frequencies_GHz = np.array([0.01, 0.1, 1, 8, 16, 25, 32, 53])
frequencies_Hz = frequencies_GHz * 1e9  # Convert to Hz

# Dielectric Loss (dB/inch)
alpha_d = 0.911 * np.sqrt(Dk) * Df * frequencies_GHz

# Conductor Loss (dB/inch)
# Skin depth (m)
omega = 2 * np.pi * frequencies_Hz
skin_depth = np.sqrt(2 / (omega * mu_0 * sigma))  # in meters
skin_depth_in = skin_depth / 0.0254  # Convert to inches

# Surface resistance (Rs)
Rs = np.sqrt(omega * mu_0 / (2 * sigma))  # ohms/square

# Base conductor loss (smooth copper)
alpha_c_smooth = (Rs / (Z_0 * W)) * 8.686  # dB/inch

# Roughness factor (Hutton model)
K = 1 + (2 / np.pi) * np.arctan(1.4 * (roughness / skin_depth)**2)
alpha_c = alpha_c_smooth * K  # Conductor loss with roughness

# Radiation Loss (negligible for stripline)
alpha_r = np.zeros_like(frequencies_GHz)

# Total Loss
alpha_total = alpha_d + alpha_c  # Radiation loss is 0
alpha_total_smooth = alpha_d + alpha_c_smooth  # Radiation loss is 0

# Plotting
plt.figure(figsize=(10, 6))
plt.semilogx(frequencies_GHz, alpha_d, label='Dielectric Loss ($\\alpha_d$)', marker='o')
plt.semilogx(frequencies_GHz, alpha_c_smooth, label='Conductor Loss (Smooth) ($\\alpha_c$)', marker='s')
plt.semilogx(frequencies_GHz, alpha_c, label='Conductor Loss (Rough RMS=1.5um)', marker='^')
plt.semilogx(frequencies_GHz, alpha_total_smooth, label='Total Loss smooth ($\\alpha_{total}$)', marker='*', linewidth=2)
plt.semilogx(frequencies_GHz, alpha_total, label='Total Loss ($\\alpha_{total}$)', marker='*', linewidth=2)

# Formatting
plt.xlabel('Frequency (GHz)')
plt.ylabel('Loss (dB/inch)')
plt.title('Stripline Loss Components for Megtron 6 (50-ohm, 5 mil width)')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()

# Show plot
plt.show()

# Print results
print("Frequency (GHz) | Dielectric Loss | Conductor Loss (Smooth) | Conductor Loss (Rough) | Total Loss")
print("-" * 80)
for i, f in enumerate(frequencies_GHz):
    print(f"{f:>10.2f}      | {alpha_d[i]:>12.4f}    | {alpha_c_smooth[i]:>15.4f}      | {alpha_c[i]:>15.4f}     | {alpha_total[i]:>10.4f}")