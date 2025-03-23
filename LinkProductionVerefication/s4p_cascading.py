import skrf as rf
#print(skrf.__version__)
import numpy as np
import matplotlib.pyplot as plt
import os

def check_s_parameter_causality(network, name, percentage_threshold=0.1):
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
    
    for i in range(n_ports):
        for j in range(n_ports):
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
            non_causal_energy = np.sum(np.abs(h_ij[t < 0])**2) * dt if np.any(t < 0) else 0
            non_causal_pct = (non_causal_energy / total_energy * 100) if total_energy > 1e-10 else 0
            non_causal_percentages[(i, j)] = non_causal_pct
            
            if non_causal_pct > percentage_threshold:
                print(f"Warning: {name} S{i+1}{j+1} violates causality (non-causal energy = {non_causal_pct:.2f}% > {percentage_threshold}%)")
    
    # Maximum non-causal percentage across all S-parameters
    max_non_causal_pct = max(non_causal_percentages.values()) if non_causal_percentages else 0
    if max_non_causal_pct <= percentage_threshold:
        print(f"{name} passes causality check (max non-causal energy={max_non_causal_pct} ≤ {percentage_threshold}%)")
    else:
        print(f"Warning: {name} violates causality (max non-causal energy={max_non_causal_pct} > {percentage_threshold}%)")
    
    return impulse_responses, non_causal_percentages, t

def check_s_parameter_passivity(network, name):
    """Check passivity for any n-port network using S^H S eigenvalues and port-specific power sums."""
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
    if np.any(max_eigenvalues > 1.0 + 1e-6):
        print(f"Warning: {name} violates passivity (max eigenvalue > 1) at some frequencies!")
    
    for port in range(n_ports):
        if np.any(port_powers[:, port] > 1.0 + 1e-6):
            print(f"Warning: {name} violates passivity at Port {port + 1} at some frequencies!")
    
    return max_eigenvalues, port_powers

def check_s_parameter_reciprocity(network, name, percentage_threshold=2.5):
    """Check reciprocity with a percentage-based threshold and return deviations per port pair."""
    n_ports = network.nports
    freq_points = len(network.f)
    TH = percentage_threshold / 100  # Convert percentage to decimal (e.g., 2.5% -> 0.025)
    
    # Dictionary to store deviations for each port pair
    reciprocity_deviations = {}
    pairs = [(j, k) for j in range(n_ports) for k in range(j + 1, n_ports)]
    
    for j, k in pairs:
        deviations = np.zeros(freq_points)
        for i in range(freq_points):
            s = network.s[i]
            delta = np.abs(s[j, k] - s[k, j])
            s_ref = max(np.abs(s[j, k]), np.abs(s[k, j]))
            if s_ref < 1e-6:
                percentage_dev = delta / 1e-6 if delta > 1e-6 else 0
            else:
                percentage_dev = (delta / s_ref) * 100
            deviations[i] = percentage_dev
        reciprocity_deviations[(j, k)] = deviations
    
    # Maximum deviation across all pairs per frequency
    max_percentage_deviation = np.max([dev for dev in reciprocity_deviations.values()], axis=0)
    
    if np.any(max_percentage_deviation > percentage_threshold):
        print(f"Warning: {name} violates reciprocity (max deviation={max_percentage_deviation} > {percentage_threshold}%) at some frequencies!")
    else:
        print(f"{name} passes reciprocity check (max deviation={max_percentage_deviation} ≤ {percentage_threshold}%)")
    
    return reciprocity_deviations, max_percentage_deviation

def plot_passivity(freq, max_eigenvalues, port_powers, name):
    """Plot passivity metrics: max eigenvalue and power sum per port."""
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
    plt.plot(freq / 1e9, max_deviation, label=f'{name} Max % Deviation', color='black', linewidth=2)
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

def cascade_s2p(file1_name, file2_name, output_name='cascaded.s2p'):
    # Set paths to current directory
    file1_path = os.path.join(os.getcwd(), file1_name)
    file2_path = os.path.join(os.getcwd(), file2_name)
    output_path = os.path.join(os.getcwd(), output_name)
    
    # Read the two S2P files
    ntwk1 = rf.Network(file1_path)
    ntwk2 = rf.Network(file2_path)
    
    # Verify frequencies match
    if not np.allclose(ntwk1.f, ntwk2.f):
        raise ValueError("Frequency points of the two networks do not match!")
    
    # Verify both are 2-port networks
    if ntwk1.nports != 2 or ntwk2.nports != 2:
        raise ValueError("Both networks must be 2-port networks!")
    
    # Cascade networks
    # Connect port 2 of file1 to port 1 of file2
    cascaded = rf.connect(ntwk1, 1, ntwk2, 0)  # port 2 (index 1) to port 1 (index 0)
    
    # Write the cascaded network to a new S2P file
    cascaded.write_touchstone(output_path)
    
    return ntwk1, ntwk2, cascaded

def cascade_s2p_or_s4p(file1_name, file2_name, output_name='cascaded'):
    file1_path = os.path.join(os.getcwd(), file1_name)
    file2_path = os.path.join(os.getcwd(), file2_name)
    output_path = os.path.join(os.getcwd(), output_name)
    
    ntwk1 = rf.Network(file1_path)
    ntwk2 = rf.Network(file2_path)
    
    # Passivity checks
    max_eig1, port_powers1 = check_s_parameter_passivity(ntwk1, file1_name)
    max_eig2, port_powers2 = check_s_parameter_passivity(ntwk2, file2_name)
    
    if not np.allclose(ntwk1.f, ntwk2.f):
        raise ValueError("Frequency points of the two networks do not match!")
    
    if ntwk1.nports != ntwk2.nports:
        raise ValueError("Both networks must have the same number of ports!")
    
    # Cascade based on number of ports
    if ntwk1.nports == 2:
        cascaded = rf.connect(ntwk1, 1, ntwk2, 0)  # S2P: Port 2 to Port 1
        output_path += '.s2p'
    elif ntwk1.nports == 4:
        cascaded = rf.connect(ntwk1, 3, ntwk2, 0)  # S4P: Port 4 to Port 1
        cascaded = rf.connect(cascaded, 2, ntwk2, 1)  # Port 3 to Port 2
        output_path += '.s2p'  # Result is 2-port
    else:
        raise ValueError("Only 2-port (S2P) or 4-port (S4P) networks are supported for cascading!")
    
    max_eig_casc, port_powers_casc = check_s_parameter_passivity(cascaded, "Cascaded Network")
    
    # Plot passivity for all networks
    plot_passivity(ntwk1.f, max_eig1, port_powers1, file1_name)
    plot_passivity(ntwk2.f, max_eig2, port_powers2, file2_name)
    plot_passivity(cascaded.f, max_eig_casc, port_powers_casc, "Cascaded Network")
    
    cascaded.write_touchstone(output_path)
    return ntwk1, ntwk2, cascaded


def plot_parameters(ntwk1, ntwk2, cascaded, ntwk3):
    freq = ntwk1.f / 1e9
    
    # Handle both 2-port and 4-port for S21 and S11 (assuming differential or single-ended appropriately)
    if ntwk1.nports == 2:
        il1 = 20 * np.log10(np.abs(ntwk1.s[:, 1, 0]))
        il2 = 20 * np.log10(np.abs(ntwk2.s[:, 1, 0]))
        il_casc = 20 * np.log10(np.abs(cascaded.s[:, 1, 0]))
        il3 = 20 * np.log10(np.abs(ntwk3.s[:, 1, 0]))
        
        rl1 = 20 * np.log10(np.abs(ntwk1.s[:, 0, 0]))
        rl2 = 20 * np.log10(np.abs(ntwk2.s[:, 0, 0]))
        rl_casc = 20 * np.log10(np.abs(cascaded.s[:, 0, 0]))
        rl3 = 20 * np.log10(np.abs(ntwk3.s[:, 0, 0]))
    else:  # 4-port, assume differential S21_dd and S11_dd
        il1 = 20 * np.log10(np.abs((ntwk1.s[:, 1, 0] - ntwk1.s[:, 1, 2] - ntwk1.s[:, 3, 0] + ntwk1.s[:, 3, 2]) / 2))
        il2 = 20 * np.log10(np.abs((ntwk2.s[:, 1, 0] - ntwk2.s[:, 1, 2] - ntwk2.s[:, 3, 0] + ntwk2.s[:, 3, 2]) / 2))
        il_casc = 20 * np.log10(np.abs(cascaded.s[:, 1, 0]))  # Cascaded is 2-port
        il3 = 20 * np.log10(np.abs((ntwk3.s[:, 1, 0] - ntwk3.s[:, 1, 2] - ntwk3.s[:, 3, 0] + ntwk3.s[:, 3, 2]) / 2))
        
        rl1 = 20 * np.log10(np.abs((ntwk1.s[:, 0, 0] - ntwk1.s[:, 0, 2] - ntwk1.s[:, 2, 0] + ntwk1.s[:, 2, 2]) / 2))
        rl2 = 20 * np.log10(np.abs((ntwk2.s[:, 0, 0] - ntwk2.s[:, 0, 2] - ntwk2.s[:, 2, 0] + ntwk2.s[:, 2, 2]) / 2))
        rl_casc = 20 * np.log10(np.abs(cascaded.s[:, 0, 0]))
        rl3 = 20 * np.log10(np.abs((ntwk3.s[:, 0, 0] - ntwk3.s[:, 0, 2] - ntwk3.s[:, 2, 0] + ntwk3.s[:, 2, 2]) / 2))
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    ax1.plot(freq, il1, label='Network 1')
    ax1.plot(freq, il2, label='Network 2')
    ax1.plot(freq, il_casc, label='Cascaded')
    ax1.plot(freq, il3, label='Network 3', linestyle='--')
    ax1.set_title('Insertion Loss (S21 or S21_dd)')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Insertion Loss (dB)')
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(freq, rl1, label='Network 1')
    ax2.plot(freq, rl2, label='Network 2')
    ax2.plot(freq, rl_casc, label='Cascaded')
    ax2.plot(freq, rl3, label='Network 3', linestyle='--')
    ax2.set_title('Return Loss (S11 or S11_dd)')
    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('Return Loss (dB)')
    ax2.grid(True)
    ax2.legend()
    
    ax3.plot(freq, il_casc, label='Cascaded IL', color='blue')
    ax3.plot(freq, il3, label='Network 3 IL', color='orange', linestyle='--')
    ax3.plot(freq, rl_casc, label='Cascaded RL', color='green')
    ax3.plot(freq, rl3, label='Network 3 RL', color='red', linestyle='--')
    ax3.set_title('Comparison: Cascaded vs Network 3')
    ax3.set_xlabel('Frequency (GHz)')
    ax3.set_ylabel('Magnitude (dB)')
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()

def main():
    # File names in current directory
    file1_name = 'Stripline_bend_Strip_bend_cutout_2mm_extended_2p5in.s2p'
    file2_name = 'Stripline_bend_Strip_bend_cutout_2mm_extended_2p5in.s2p'
    file3_name = 'Stripline_bend_Strip_bend_cutout_2mm_extended_5in.s2p'
    
    try:
        # Cascade the networks
        ntwk1, ntwk2, cascaded = cascade_s2p_or_s4p(file1_name, file2_name, output_name='cascaded')
        # Read the third S2P file for comparison
        file3_path = os.path.join(os.getcwd(), file3_name)
        ntwk3 = rf.Network(file3_path)
        
        max_eig1, port_powers1 = check_s_parameter_passivity(ntwk1, file1_name)
        plot_passivity(ntwk1.f, max_eig1, port_powers1, file1_name)
        max_eig2, port_powers2 = check_s_parameter_passivity(ntwk2, file2_name)
        plot_passivity(ntwk2.f, max_eig2, port_powers2, file2_name)
        max_eig3, port_powers3 = check_s_parameter_passivity(ntwk3, file3_name)
        plot_passivity(ntwk3.f, max_eig3, port_powers3, file3_name)

        reciprocity_dev1, max_dev1 = check_s_parameter_reciprocity(ntwk1, file1_name)
        plot_reciprocity(ntwk1.f, reciprocity_dev1, max_dev1, file1_name)
        reciprocity_dev2, max_dev2 = check_s_parameter_reciprocity(ntwk2, file2_name)
        plot_reciprocity(ntwk2.f, reciprocity_dev2, max_dev2, file2_name)
        reciprocity_dev3, max_dev3 = check_s_parameter_reciprocity(ntwk3, file3_name)
        plot_reciprocity(ntwk3.f, reciprocity_dev3, max_dev3, file3_name)
        reciprocity_cascad, max_dev4 = check_s_parameter_reciprocity(cascaded, "cascaded")
        plot_reciprocity(cascaded.f, reciprocity_cascad, max_dev4, "cascaded")

        # Causality checks and plots
        h1, nc_pct1, t1 = check_s_parameter_causality(ntwk1, file1_name)
        plot_causality(t1, h1, file1_name)
        h2, nc_pct2, t2 = check_s_parameter_causality(ntwk2, file2_name)
        plot_causality(t2, h2, file2_name)
        h3, nc_pct3, t3 = check_s_parameter_causality(ntwk3, file3_name)
        plot_causality(t3, h3, file3_name)
        h_casc, nc_pct_casc, t_casc = check_s_parameter_causality(cascaded, "cascaded")
        plot_causality(t_casc, h_casc, "cascaded")        
        # Verify frequency compatibility with cascaded result
        if not np.allclose(cascaded.f, ntwk3.f):
            raise ValueError("Frequency points of cascaded network and Network 3 do not match!")
        
        # Plot parameters
        plot_parameters(ntwk1, ntwk2, cascaded, ntwk3)
        plt.show()
        print("Cascading completed successfully!")
        print(f"Cascaded S-parameters saved to 'cascaded.s2p' in {os.getcwd()}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
