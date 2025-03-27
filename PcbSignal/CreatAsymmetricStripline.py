import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
# Constants
SPEED_OF_LIGHT = 299792458  # Exact speed of light in vacuum (m/s)
CONDUCTIVITY = 5.8e7  # Conductivity of copper (S/m)
MU_0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)

# Material properties table (dielectric constant and loss tangent)
MATERIALS = {
    "FR-4": {"er": 4.5, "loss_tangent": 0.02},
    "Rogers RO4003C": {"er": 3.55, "loss_tangent": 0.0027},
    "Rogers RO4350B": {"er": 3.66, "loss_tangent": 0.0037},
    "Isola I-Tera MT40": {"er": 3.45, "loss_tangent": 0.003},
    "Nelco N4000-13": {"er": 3.7, "loss_tangent": 0.009},
    "Teflon (PTFE)": {"er": 2.1, "loss_tangent": 0.0002},
    "Megtron 6": {"er": 3.7, "loss_tangent": 0.002},
}

# Function to convert mils to meters
def mil_to_meters(mils):
    return mils * 0.0254e-3  # 1 mil = 0.0254 mm = 0.0254e-3 meters

# Function to calculate the characteristic impedance for an asymmetric stripline
def asymmetric_stripline_impedance(width, thickness, h1, h2, dielectric_constant):
    effective_dielectric_constant = dielectric_constant  # For stripline, it's the same as the substrate
    b = h1 + h2 + thickness  # Total dielectric thickness
    w_eff = width + (0.35 * thickness)  # Effective trace width
    Z0 = (60 / np.sqrt(effective_dielectric_constant)) * np.log(
        (4 * b) / (0.67 * np.pi * (0.8 * w_eff + thickness))
    )
    return Z0, effective_dielectric_constant

# New function for differential stripline impedance
def asymmetric_differential_stripline_impedance(width, thickness, h1, h2, spacing, dielectric_constant):
    effective_dielectric_constant = dielectric_constant
    b = h1 + h2 + thickness
    w_eff = width + (0.35 * thickness)
    Z0_single = (60 / np.sqrt(effective_dielectric_constant)) * np.log(
        (4 * b) / (0.67 * np.pi * (0.8 * w_eff + thickness))
    )
    Z_odd = Z0_single * (1 - 0.48 * np.exp(-0.96 * spacing / b))
    Z_diff = 2 * Z_odd
    return Z_diff, effective_dielectric_constant

# Function to calculate the propagation constant (including losses)
def calculate_propagation_constant(frequencies, effective_dielectric_constant, loss_tangent, trace_width, trace_thickness, conductivity, Z0):
    omega = 2 * np.pi * frequencies
    c = SPEED_OF_LIGHT
    skin_depth = np.sqrt(2 / (omega * MU_0 * conductivity))
    R = 1 / (conductivity * skin_depth * (trace_width + trace_thickness))
    alpha_c = R / (2 * Z0)
    alpha_d = (omega * np.sqrt(effective_dielectric_constant)) / (2 * c) * loss_tangent
    alpha = alpha_c + alpha_d
    alpha_dB_per_inch = alpha * 8.685889638 / 39.3701
    beta = omega * np.sqrt(effective_dielectric_constant) / c
    beta_per_inch = beta / 39.3701
    gamma = alpha + 1j * beta
    return gamma, alpha_dB_per_inch, beta_per_inch

# Function to update dielectric constant and loss tangent based on material selection
def update_material(event):
    selected_material = material_combobox.get()
    if selected_material != "manual":
        entry_er.delete(0, tk.END)
        entry_er.insert(0, MATERIALS[selected_material]["er"])
        entry_loss_tangent.delete(0, tk.END)
        entry_loss_tangent.insert(0, MATERIALS[selected_material]["loss_tangent"])

# Function to plot impedance sensitivity
def plot_impedance_sensitivity(trace_width, trace_thickness, h1, h2, dielectric_constant, plot_frame):
    percentages = np.arange(-10, 15, 5)
    Z0_width, Z0_thickness, Z0_h1, Z0_h2, Z0_er = [], [], [], [], []
    
    for p in percentages:
        width = trace_width * (1 + p / 100)
        Z0, _ = asymmetric_stripline_impedance(width, trace_thickness, h1, h2, dielectric_constant)
        Z0_width.append(Z0)
        
        thickness = trace_thickness * (1 + p / 100)
        Z0, _ = asymmetric_stripline_impedance(trace_width, thickness, h1, h2, dielectric_constant)
        Z0_thickness.append(Z0)
        
        h1_sweep = h1 * (1 + p / 100)
        Z0, _ = asymmetric_stripline_impedance(trace_width, trace_thickness, h1_sweep, h2, dielectric_constant)
        Z0_h1.append(Z0)
        
        h2_sweep = h2 * (1 + p / 100)
        Z0, _ = asymmetric_stripline_impedance(trace_width, trace_thickness, h1, h2_sweep, dielectric_constant)
        Z0_h2.append(Z0)
        
        er_sweep = dielectric_constant * (1 + p / 100)
        Z0, _ = asymmetric_stripline_impedance(trace_width, trace_thickness, h1, h2, er_sweep)
        Z0_er.append(Z0)
    
    # Clear previous plot
    for widget in plot_frame.winfo_children():
        widget.destroy()
    
    # Set figure size to 50% of window width (1000 pixels initially)
    dpi = 100
    fig_width = 5  # 500 pixels / 100 dpi = 5 inches (50% of 1000px window)
    fig_height = 4
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.plot(percentages, Z0_width, label="Trace Width")
    ax.plot(percentages, Z0_thickness, label="Trace Thickness")
    ax.plot(percentages, Z0_h1, label="Dielectric Thickness to Plane 1")
    ax.plot(percentages, Z0_h2, label="Dielectric Thickness to Plane 2")
    ax.plot(percentages, Z0_er, label="Dielectric Constant")
    ax.set_xlabel("Percentage Change (%)")
    ax.set_ylabel("Characteristic Impedance (Z0) [ohms]")
    ax.set_title("Impedance Sensitivity Analysis")
    ax.legend()
    ax.grid()
    
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.X)  # Fill horizontally, not vertically

def plot_diff_impedance_sensitivity(trace_width, trace_thickness, h1, h2, spacing, dielectric_constant, diff_plot_frame):
    percentages = np.arange(-10, 15, 5)
    Z_diff_width, Z_diff_thickness, Z_diff_h1, Z_diff_h2, Z_diff_er, Z_spacing = [], [], [], [], [], []
    
    for p in percentages:
        width = trace_width * (1 + p / 100)
        Z_diff, _ = asymmetric_differential_stripline_impedance(width, trace_thickness, h1, h2, spacing, dielectric_constant)
        Z_diff_width.append(Z_diff)
        
        thickness = trace_thickness * (1 + p / 100)
        Z_diff, _ = asymmetric_differential_stripline_impedance(trace_width, thickness, h1, h2, spacing, dielectric_constant)
        Z_diff_thickness.append(Z_diff)
        
        h1_sweep = h1 * (1 + p / 100)
        Z_diff, _ = asymmetric_differential_stripline_impedance(trace_width, trace_thickness, h1_sweep, h2, spacing, dielectric_constant)
        Z_diff_h1.append(Z_diff)
        
        h2_sweep = h2 * (1 + p / 100)
        Z_diff, _ = asymmetric_differential_stripline_impedance(trace_width, trace_thickness, h1, h2_sweep, spacing, dielectric_constant)
        Z_diff_h2.append(Z_diff)
        
        er_sweep = dielectric_constant * (1 + p / 100)
        Z_diff, _ = asymmetric_differential_stripline_impedance(trace_width, trace_thickness, h1, h2, spacing, er_sweep)
        Z_diff_er.append(Z_diff)

        spacing_sweep = spacing * (1 + p / 100)
        Z_diff, _ = asymmetric_differential_stripline_impedance(trace_width, trace_thickness, h1, h2, spacing_sweep, dielectric_constant)
        Z_spacing.append(Z_diff)
    
    # Clear previous plot
    for widget in diff_plot_frame.winfo_children():
        widget.destroy()
    
    # Set figure size to 50% of window width (assuming 1200px window)
    dpi = 100
    fig_width = 6  # 600 pixels / 100 dpi = 6 inches (50% of 1200px window)
    fig_height = 4
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.plot(percentages, Z_diff_width, label="Trace Width")
    ax.plot(percentages, Z_diff_thickness, label="Trace Thickness")
    ax.plot(percentages, Z_diff_h1, label="Dielectric Thickness to Plane 1")
    ax.plot(percentages, Z_diff_h2, label="Dielectric Thickness to Plane 2")
    ax.plot(percentages, Z_diff_er, label="Dielectric Constant")
    ax.plot(percentages, Z_spacing, label="spacing")
    ax.set_xlabel("Percentage Change (%)")
    ax.set_ylabel("Differential Impedance (Z_diff) [ohms]")
    ax.set_title("Differential Impedance Sensitivity Analysis")
    ax.legend()
    ax.grid()
    
    canvas = FigureCanvasTkAgg(fig, master=diff_plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.X)

# Function to run the calculation and display results
def run_calculation():
    try:
        trace_width = mil_to_meters(float(entry_width.get()))
        trace_thickness_oz = float(entry_thickness.get())
        trace_thickness = mil_to_meters(trace_thickness_oz * 1.37)
        dielectric_thickness_1 = mil_to_meters(float(entry_h1.get()))
        dielectric_thickness_2 = mil_to_meters(float(entry_h2.get()))
        trace_length = float(entry_length.get()) * 0.0254
        dielectric_constant = float(entry_er.get())
        loss_tangent = float(entry_loss_tangent.get())
        
        Z0, effective_dielectric_constant = asymmetric_stripline_impedance(
            trace_width, trace_thickness, dielectric_thickness_1, dielectric_thickness_2, dielectric_constant
        )
        # Inside run_calculation
        spacing = mil_to_meters(float(entry_spacing.get()))
        Z_diff, Z_diff_effective_dielectric_constant = asymmetric_differential_stripline_impedance(
            trace_width, trace_thickness, dielectric_thickness_1, dielectric_thickness_2, spacing, dielectric_constant
        )
        
        frequencies = np.linspace(0, 60e9, 6000)
        gamma, alpha_dB_per_inch, beta_per_inch = calculate_propagation_constant(
            frequencies, effective_dielectric_constant, loss_tangent, trace_width, trace_thickness, CONDUCTIVITY, Z0
        )
        
        trace_network = rf.Network()
        trace_network.frequency = rf.Frequency.from_f(frequencies, unit='Hz')
        trace_network.s = np.zeros((len(frequencies), 2, 2), dtype=complex)
        print(f"Characteristic Impedance (Z0): {Z0:.2f} ohms\n")
        for i, f in enumerate(frequencies):
            A = np.cosh(gamma[i] * trace_length)
            B = Z0 * np.sinh(gamma[i] * trace_length)
            C = (1 / Z0) * np.sinh(gamma[i] * trace_length)
            D = np.cosh(gamma[i] * trace_length)
            denom = A + B / Z0 + C * Z0 + D
            trace_network.s[i, 0, 0] = (A + B / Z0 - C * Z0 - D) / denom
            trace_network.s[i, 0, 1] = 2 / denom
            trace_network.s[i, 1, 0] = 2 / denom
            trace_network.s[i, 1, 1] = (-A + B / Z0 - C * Z0 + D) / denom
        
        target_frequencies = [1e9, 4e9, 8e9, 13.28125e9, 16e9, 26.5625e9]
        insertion_loss = [20 * np.log10(np.abs(trace_network.s[np.argmin(np.abs(frequencies - f)), 1, 0])) for f in target_frequencies]
        
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"Characteristic Impedance (Z0): {Z0:.2f} ohms\n")
        idx_1GHz = np.argmin(np.abs(frequencies - 1e9))
        result_text.insert(tk.END, f"Attenuation Constant at 1 GHz: {alpha_dB_per_inch[idx_1GHz]:.4f} dB/inch\n")
        result_text.insert(tk.END, f"Phase Constant at 1 GHz: {beta_per_inch[idx_1GHz]:.4f} rad/inch\n\n")
        result_text.insert(tk.END, "Insertion Loss (S21) at:\n")
        for freq, loss in zip(["1 GHz", "4 GHz", "8 GHz", "13.28125 GHz", "16 GHz", "26.5625 GHz"], insertion_loss):
            result_text.insert(tk.END, f"{freq}: {loss:.2f} dB\n")
        
        # Differential network (4-port)
        gamma_diff, alpha_dB_per_inch_diff, beta_per_inch_diff = calculate_propagation_constant(
            frequencies, Z_diff_effective_dielectric_constant, loss_tangent, trace_width, trace_thickness, CONDUCTIVITY, Z_diff
        )
        trace_network_diff = rf.Network()
        trace_network_diff.frequency = rf.Frequency.from_f(frequencies, unit='Hz')
        trace_network_diff.s = np.zeros((len(frequencies), 4, 4), dtype=complex)
        for i, f in enumerate(frequencies):
            A = np.cosh(gamma_diff[i] * trace_length)
            B = Z_diff * np.sinh(gamma_diff[i] * trace_length)
            C = (1 / Z_diff) * np.sinh(gamma_diff[i] * trace_length)
            D = np.cosh(gamma_diff[i] * trace_length)
            denom = A + B / Z_diff + C * Z_diff + D
            # Simplified 4-port S-parameters for differential pair (assuming symmetry and no cross-coupling between pairs)
            S11 = (A + B / Z_diff - C * Z_diff - D) / denom
            S21 = 2 / denom
            # Port 1 (input +), Port 2 (output +), Port 3 (input -), Port 4 (output -)
            trace_network_diff.s[i, 0, 0] = S11  # S11 (port 1 to port 1)
            trace_network_diff.s[i, 0, 1] = S21  # S12 (port 1 to port 2)
            trace_network_diff.s[i, 1, 0] = S21  # S21 (port 2 to port 1)
            trace_network_diff.s[i, 1, 1] = S11  # S22 (port 2 to port 2)
            trace_network_diff.s[i, 2, 2] = S11  # S33 (port 3 to port 3)
            trace_network_diff.s[i, 2, 3] = S21  # S34 (port 3 to port 4)
            trace_network_diff.s[i, 3, 2] = S21  # S43 (port 4 to port 3)
            trace_network_diff.s[i, 3, 3] = S11  # S44 (port 4 to port 4)
            # Differential assumes minimal coupling between + and - traces for simplicity

        # Differential results
        insertion_loss_diff = [20 * np.log10(np.abs(trace_network_diff.s[np.argmin(np.abs(frequencies - f)), 1, 0])) for f in target_frequencies]
        result_text_diff.delete(1.0, tk.END)
        result_text_diff.insert(tk.END, f"Differential Impedance (Z_diff): {Z_diff:.2f} ohms\n")
        idx_16GHz = np.argmin(np.abs(frequencies - 16e9))
        result_text_diff.insert(tk.END, f"Attenuation Constant at 16 GHz: {alpha_dB_per_inch_diff[idx_16GHz]:.4f} dB/inch\n")
        result_text_diff.insert(tk.END, f"Phase Constant at 16 GHz: {beta_per_inch_diff[idx_16GHz]:.4f} rad/inch\n\n")
        result_text_diff.insert(tk.END, "Differential Insertion Loss (S21) at:\n")
        for freq, loss in zip(["1 GHz", "4 GHz", "8 GHz", "13.28125 GHz", "16 GHz", "26.5625 GHz"], insertion_loss_diff):
            result_text_diff.insert(tk.END, f"{freq}: {loss:.2f} dB\n")
        
        print("trace_network.write_touchstone\n")
        trace_network.write_touchstone("Asymmetric_stripline_with_losses.s2p")
        trace_network_diff.write_touchstone("asymmetric_differential_stripline_with_losses.s4p")
        plot_impedance_sensitivity(trace_width, trace_thickness, dielectric_thickness_1, dielectric_thickness_2, dielectric_constant, plot_frame)
        # Plot differential impedance sensitivity
        plot_diff_impedance_sensitivity(trace_width, trace_thickness, dielectric_thickness_1, dielectric_thickness_2, spacing, dielectric_constant, diff_plot_frame)
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create the GUI
root = tk.Tk()
root.title("Asymmetric Stripline Calculator")
root.geometry("1200x920")
# Configure column widths
root.grid_columnconfigure(0, minsize=400, weight=1)  # Input frame (left)
root.grid_columnconfigure(1, minsize=200, weight=1)  # Material frame (middle)
root.grid_columnconfigure(2, minsize=300, weight=1)  # Diff frame (right)
# Input fields
input_frame = tk.Frame(root)
input_frame.grid(row=0, column=0, sticky="nw", padx=7, pady=10)

tk.Label(input_frame, text="Trace Width (mils):").grid(row=0, column=0, sticky="w")
entry_width = tk.Entry(input_frame, width=20)
entry_width.insert(0, "4")
entry_width.grid(row=0, column=1, sticky="w")

tk.Label(input_frame, text="Trace Thickness (oz):").grid(row=1, column=0, sticky="w")
entry_thickness = ttk.Combobox(input_frame, values=[0.5, 1, 1.5, 2, 2.5, 3], width=17)
entry_thickness.grid(row=1, column=1, sticky="w")
entry_thickness.current(1)

tk.Label(input_frame, text="Dielectric Thickness to Plane 1 (mils):").grid(row=2, column=0, sticky="w")
entry_h1 = tk.Entry(input_frame, width=20)
entry_h1.insert(0, "4")
entry_h1.grid(row=2, column=1, sticky="w")

tk.Label(input_frame, text="Dielectric Thickness to Plane 2 (mils):").grid(row=3, column=0, sticky="w")
entry_h2 = tk.Entry(input_frame, width=20)
entry_h2.insert(0, "3.5")
entry_h2.grid(row=3, column=1, sticky="w")

tk.Label(input_frame, text="Trace Length (inches):").grid(row=4, column=0, sticky="w")
entry_length = tk.Entry(input_frame, width=20)
entry_length.insert(0, "5")
entry_length.grid(row=4, column=1, sticky="w")

tk.Label(input_frame, text="Dielectric Constant (εr):").grid(row=5, column=0, sticky="w")
entry_er = tk.Entry(input_frame, width=20)
entry_er.grid(row=5, column=1, sticky="w")

tk.Label(input_frame, text="Loss Tangent (tan δ):").grid(row=6, column=0, sticky="w")
entry_loss_tangent = tk.Entry(input_frame, width=20)
entry_loss_tangent.grid(row=6, column=1, sticky="w")

# Material selection
material_frame = tk.Frame(root)
material_frame.grid(row=0, column=1, sticky="nw", padx=7, pady=10)

tk.Label(material_frame, text="Material:").grid(row=0, column=0, sticky="e")
material_combobox = ttk.Combobox(material_frame, values=["manual"] + list(MATERIALS.keys()), width=17)
material_combobox.grid(row=0, column=0, sticky="n")
material_combobox.current(0)
material_combobox.bind("<<ComboboxSelected>>", update_material)

# New frame for Spacing Between Traces (right-aligned)
diff_frame = tk.Frame(root)
diff_frame.grid(row=0, column=2, sticky="nw", padx=7, pady=10)

tk.Label(diff_frame, text="diff Traces width (mils):").grid(row=1, column=0, sticky="ne")
entry_diff_width = tk.Entry(diff_frame, width=20)
entry_diff_width.insert(0, "10")  # Default value
entry_diff_width.grid(row=1, column=1, sticky="e")

tk.Label(diff_frame, text="Spacing Between Traces (mils):").grid(row=2, column=0, sticky="ne")
entry_spacing = tk.Entry(diff_frame, width=20)
entry_spacing.insert(0, "10")  # Default value
entry_spacing.grid(row=2, column=1, sticky="e")


# Run button
run_button = tk.Button(root, text="Run Calculation", command=run_calculation, bg="green", fg="white", width=20)
run_button.grid(row=1, column=0, columnspan=2, pady=10)

# Result display
result_frame = tk.Frame(root)
result_frame.grid(row=2, column=0, sticky="nw", padx=10, pady=10)

result_text = tk.Text(result_frame, height=15, width=60)
result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# 2nd Result display
result_frame_diff = tk.Frame(root)
result_frame_diff.grid(row=2, column=2, sticky="ne", padx=10, pady=10)

result_text_diff = tk.Text(result_frame_diff, height=15, width=60)
result_text_diff.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Plot frame (below result_frame)
plot_frame = tk.Frame(root)
plot_frame.grid(row=3, column=0, sticky="nw", padx=7, pady=10)

# Image frame (between plots)
image_frame = tk.Frame(root)
image_frame.grid(row=3, column=1, sticky="n", padx=7, pady=10)
try:
    # Load and resize the image (optional: adjust size as needed)
    image = Image.open("logo.jpg")
    image = image.resize((150, 150), Image.Resampling.LANCZOS)  # Resize to 150x150 pixels
    photo = ImageTk.PhotoImage(image)
    image_label = tk.Label(image_frame, image=photo)
    image_label.image = photo  # Keep a reference to avoid garbage collection
    image_label.pack()
except Exception as e:
    print(f"Error loading image: {e}")
    tk.Label(image_frame, text="Image not found").pack()

# Differential plot frame
diff_plot_frame = tk.Frame(root)
diff_plot_frame.grid(row=3, column=2, sticky="ne", padx=7, pady=10)

# Start the GUI
root.mainloop()