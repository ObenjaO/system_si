import argparse
import skrf as rf
import numpy as np
from scipy.ndimage import uniform_filter1d  # For efficient moving average
import matplotlib.pyplot as plt
import os
import sys
from Plot_and_compare import plot_s_parameters
# How to run the file:
# python script.py "/path/to/your/file.txt" True
# python script.py --help


def check_s_parameter_causality(network, name, percentage_threshold=0.1,port_assignment="12>>34"):
    """Check causality using IFFT and compute non-causal energy percentage."""
    n_ports = network.nports
    freq = network.f
    N = len(freq)
    df = freq[1] - freq[0] if N > 1 else 1.0  # Frequency step, default to 1 if single point
    
    # Pad frequency data for better time resolution
    N_pad = 2 * N
    t = np.fft.fftshift(np.fft.fftfreq(N_pad, d=df))  # Time axis based on padded length
    dt = t[1] - t[0] if len(t) > 1 else 1.0
    
    # Dictionary to store impulse responses and non-causal percentages
    impulse_responses = {}
    non_causal_percentages = {}
    # Define the port pairs based on port_assignment
    if port_assignment == "12>>34":
    # P1/P2 -> P3/P4 (Differential Pair Convention)
        port_pairs = [[0, 2], [1, 3]]  # [1,3] and [2,4] in 1-based indexing
    elif port_assignment == "13>>24":
    # P1/P3 -> P2/P4 (Interleaved Differential Assignment)
        port_pairs = [[0, 1], [2, 3]]  # [1,2] and [3,4] in 1-based indexing
    elif port_assignment == "1>>2":
    # P1/P3 -> P2/P4 (Interleaved Differential Assignment)
        port_pairs = [[1, 2]]  # [1,2] and [3,4] in 1-based indexing
    else:
        raise ValueError(f"Unsupported port_assignment: {port_assignment}")

    # Iterate over the specified port pairs
    for i, j in port_pairs:
        # Extrapolate S_ij to DC (simple linear from first point)
        s_ij = network.s[:, i, j]
        if freq[0] > 0:
            s_dc = s_ij[0] - (s_ij[1] - s_ij[0]) * freq[0] / (freq[1] - freq[0])
            s_extended = np.concatenate(([s_dc], s_ij))
        else:
            s_extended = s_ij
        
        # Zero-pad to N_pad
        s_padded = np.pad(s_extended, (0, N_pad - len(s_extended)), mode='constant')
        
        # Compute impulse response via IFFT
        h_ij = np.fft.ifftshift(np.fft.ifft(s_padded))
        impulse_responses[(i, j)] = h_ij
        
        # Compute energies
        total_energy = np.sum(np.abs(h_ij)**2) * dt
        # below possible fix if floating-point issues arise
        # non_causal_mask = t < -1e-12  # Small tolerance to avoid floating-point issues
        # non_causal_energy = np.sum(np.abs(h_ij[non_causal_mask])**2) * dt

        non_causal_energy = np.sum(np.abs(h_ij[t < 0])**2) * dt if np.any(t < 0) else 0
        non_causal_pct = (100 * non_causal_energy / total_energy) if total_energy < 1e-10 else 0
        non_causal_percentages[(i, j)] = non_causal_pct
        #print(f"Non-causal energy percentage for {name} S{i+1}{j+1}: {non_causal_energy:.2e} X {total_energy:.2e} ==> {non_causal_pct:.3f}")
        if non_causal_pct > percentage_threshold:
            print(f"Warning: {name} S{i+1}{j+1} violates causality (non-causal energy = {non_causal_pct:.2f}% > {percentage_threshold}%)")

        # Maximum non-causal percentage across all S-parameters
    max_non_causal_pct = max(non_causal_percentages.values()) if non_causal_percentages else 0
        #if max_non_causal_pct <= percentage_threshold:
        #    print(f"{name} passes causality check (max non-causal energy={max_non_causal_pct} ≤ {percentage_threshold}%)")
        #else:
        #    print(f"Warning: {name} violates causality (max non-causal energy={max_non_causal_pct} > {percentage_threshold}%)")
        
    return impulse_responses, non_causal_percentages, t

def check_s_parameter_passivity(network, name):
    """Check passivity for any n-port network using S^H S eigenvalues and port-specific power sums."""
    PASSIVITY_TOL = 1e-6
    n_ports = network.nports
    freq_points = len(network.f)
    
    # Arrays for results
    max_eigenvalues = np.zeros(freq_points)  # Max eigenvalue of S^H S
    port_powers = np.zeros((freq_points, n_ports))  # Power sum for each port
    
    # Compute S^H S and port-specific power sums for each frequency
    for i in range(freq_points):
        s = network.s[i]  # S-matrix at this frequency
        s_h = np.conjugate(s.T)  # Hermitian (conjugate transpose)
        s_h_s = np.dot(s_h, s)  # S^H S
        eigenvalues = np.linalg.eigvals(s_h_s)  # Compute eigenvalues
        max_eigenvalues[i] = np.max(np.abs(eigenvalues))  # Maximum absolute eigenvalue
        
        # Power sum for each port (sum of |S_ij|^2 over j for fixed i)
        for port in range(n_ports):
            port_powers[i, port] = np.sum(np.abs(s[port, :])**2)
    
    # Check passivity
    if np.any(max_eigenvalues > 1.0 + PASSIVITY_TOL):
        print(f"Warning: {name} violates passivity (max eigenvalue > 1) at some frequencies!")
    
    for port in range(n_ports):
        if np.any(port_powers[:, port] > 1.0 + PASSIVITY_TOL):
            print(f"Warning: {name} violates passivity at Port {port + 1} at some frequencies!")
    
    return max_eigenvalues, port_powers

def check_s_parameter_reciprocity(network, name, percentage_threshold=2.5):
    """Check reciprocity with a percentage-based threshold and return deviations per port pair."""
    base_window_size=1
    n_ports = network.nports
    freq_points = len(network.f)
    freq_ghz = network.f / 1e9  # Convert frequency to GHz
    TH = percentage_threshold / 100  # Convert percentage to decimal (e.g., 2.5% -> 0.025)
    
    # Smooth S-parameters along the frequency axis (shape: [ports, ports, freq])
    smoothed_s = np.zeros_like(network.s)
    # Smooth each S-parameter with dynamic window
    for j in range(n_ports):
        for k in range(n_ports):
            for i in range(freq_points):
                # Calculate dynamic window size (base + 1 per GHz)
                window_size = base_window_size + 3*int(freq_ghz[i])
                
                # Adjust window size at edges
                half_window = window_size // 2
                start = max(0, i - half_window)
                end = min(freq_points, i + half_window + 1)
                effective_window = end - start
                
                # Apply moving average to magnitude
                smoothed_s[i, j, k] = np.mean(np.abs(network.s[start:end, j, k]))
            
    
    # Check reciprocity on smoothed data
    reciprocity_deviations = {}
    pairs = [(j, k) for j in range(n_ports) for k in range(j + 1, n_ports)]
    
    for j, k in pairs:
        deviations = np.zeros(freq_points)
        for i in range(freq_points):
            s = smoothed_s[i]
            delta = np.abs(s[j, k] - s[k, j])
            s_ref = max(np.abs(s[j, k]), np.abs(s[k, j]))
            if s_ref < 1e-2:
                percentage_dev = delta / 1e-2 if delta > 1e-2 else 0
            else:
                percentage_dev = (delta / s_ref) * 100
            deviations[i] = percentage_dev
        reciprocity_deviations[(j, k)] = deviations
    
    max_percentage_deviation = np.max([dev for dev in reciprocity_deviations.values()], axis=0)
    
    if np.any(max_percentage_deviation > percentage_threshold):
        print(f"Warning: {name} violates reciprocity (max deviation={np.max(max_percentage_deviation):.2f}% > {percentage_threshold}%)")

    return reciprocity_deviations, max_percentage_deviation

def plot_passivity(freq, max_eigenvalues, port_powers, name):
    """Plot passivity metrics: max eigenvalue and power sum per port."""
    #print(f"plot_passivity::Max eigenvalues: {max(max_eigenvalues)}")
    n_ports = port_powers.shape[1]
    
    plt.figure(figsize=(10, 6))
    plt.plot(freq / 1e9, max_eigenvalues, label=f'{name} Max Eigenvalue of S^H S', color='black', linewidth=2)
    
    colors = ['blue', 'green', 'orange', 'purple']  # Extend if more ports needed
    for port in range(n_ports):
        plt.plot(freq / 1e9, port_powers[:, port], label=f'{name} Port {port + 1} Power Sum', 
                 color=colors[port % len(colors)], linestyle='--')
    
    plt.axhline(y=1.0, color='r', linestyle='--', label='Passivity Limit')
    plt.title(f'Passivity Check: {name}')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Passivity Metric')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.show()

def plot_reciprocity(freq, reciprocity_deviations, max_deviation, name, percentage_threshold=2.5):
    """Plot reciprocity percentage deviation per port pair and max deviation."""
    plt.figure(figsize=(12, 8))
    
    # Plot each port pair
    colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
    for idx, ((j, k), deviations) in enumerate(reciprocity_deviations.items()):
        plt.plot(freq / 1e9, deviations, label=f'{name} S{j+1}{k+1} vs S{k+1}{j+1}', 
                 color=colors[idx % len(colors)], linestyle='--')
    
    # Plot max deviation
    #plt.plot(freq / 1e9, max_deviation, label=f'{name} Max % Deviation', color='black', linewidth=2)
    plt.axhline(y=percentage_threshold, color='r', linestyle='--', label=f'Threshold ({percentage_threshold}%)')
    
    plt.title(f'Reciprocity Check: {name}')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Percentage Deviation (%)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

def plot_causality(t, impulse_responses, name, percentage_threshold=0.1):
    """Plot impulse responses for each S-parameter with t=0 line."""
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'pink']
    for idx, ((i, j), h_ij) in enumerate(impulse_responses.items()):
        plt.plot(t * 1e9, np.abs(h_ij), label=f'{name} |h{i+1}{j+1}(t)|', 
                 color=colors[idx % len(colors)], linestyle='-')
    
    plt.axvline(x=0, color='r', linestyle='--', label='t = 0')
    plt.title(f'Impulse Response: {name}')
    plt.xlabel('Time (ns)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

def main():
    res = True
    PASSIVITY_TOL = 1e-6

    parser = argparse.ArgumentParser(description="Test S parameter passivity, reciprocity & causality.")
    
    # Adding arguments for file path and debug mode
    parser.add_argument('file', type=str, help="The full path to the file to process.")
    parser.add_argument('debug', type=bool, help="Enable or disable debug mode (True/False).")
    
    # Parsing the command-line arguments
    args = parser.parse_args()
    file = sys.argv[1]
    debug_mode = sys.argv[2]

    #print(f"File: {file}")
    #print(f"Debug raw: {repr(debug_mode)} (type: {type(debug_mode)})")  # Print type
    # Check if we got exactly 2 arguments
    if len(vars(args)) != 2:
        print("Error: Expected 2 arguments (file path and debug mode).")
        return
    
    # Check that the first argument is a string (it is because we specified 'type=str')
    file_path = args.file
    if not isinstance(file_path, str):
        print("Error: The file path should be a string.")
        return
    
    # Check that the second argument is either 'True' or 'False'
    debug_mode = debug_mode.strip()
    if debug_mode not in ["True", "False"]:
        print("Error: The debug mode should be either 'True' or 'False'. Received: {repr(debug_mode)}")
        print(debug_mode)
        return
    
    # Extracting the file name
    file_name = os.path.basename(args.file)
    
    ntwk = rf.Network(args.file)
    # Convert the 'debug' string to boolean
    debug_mode = debug_mode == "True"

    # ======================== Test starts here ========================
    # Passivity checks and plots    
    max_eig, port_powers = check_s_parameter_passivity(ntwk, file_name)
    if debug_mode:
        print(f"Max eigenvalues: {max(max_eig)}")
        plot_passivity(ntwk.f, max_eig, port_powers, file_name)
    
    if np.any(max_eig > 1.0 + PASSIVITY_TOL):
        res = False

    reciprocity_dev, max_dev = check_s_parameter_reciprocity(ntwk, file_name)
    if debug_mode:
        print(f"Max reciprocity deviations: {max(max_dev)}")
        plot_reciprocity(ntwk.f, reciprocity_dev, max_dev, file_name)
        plot_s_parameters(ntwk)
    if np.any(max_dev > 2.5):
        res = False
    # Causality checks and plots
    h1, nc_pct1, t1 = check_s_parameter_causality(ntwk, file_name)
    if debug_mode:
        print(f"Non-causal percentages: {nc_pct1}")
        plot_causality(t1, h1, file_name)
        plt.show()
    if (max(nc_pct1.values()) > 0.1):
        res = False
    # ======================== Test ends here ========================

    return res

if __name__ == "__main__":
    main()
    