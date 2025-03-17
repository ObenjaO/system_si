import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
import os


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
    plot_passivity(ntwk1.f, max_eig2, port_powers2, file2_name)
    plot_passivity(ntwk1.f, max_eig_casc, port_powers_casc, "Cascaded Network")
    
    cascaded.write_touchstone(output_path)
    return ntwk1, ntwk2, cascaded
'''
def plot_parameters(ntwk1, ntwk2, cascaded, ntwk3):
    # Calculate insertion loss (S21) and return loss (S11)
    freq = ntwk1.f / 1e9  # Convert to GHz
    
    il1 = 20 * np.log10(np.abs(ntwk1.s[:, 1, 0]))  # S21 in dB
    il2 = 20 * np.log10(np.abs(ntwk2.s[:, 1, 0]))
    il_casc = 20 * np.log10(np.abs(cascaded.s[:, 1, 0]))
    il3 = 20 * np.log10(np.abs(ntwk3.s[:, 1, 0]))
    
    rl1 = 20 * np.log10(np.abs(ntwk1.s[:, 0, 0]))  # S11 in dB
    rl2 = 20 * np.log10(np.abs(ntwk2.s[:, 0, 0]))
    rl_casc = 20 * np.log10(np.abs(cascaded.s[:, 0, 0]))
    rl3 = 20 * np.log10(np.abs(ntwk3.s[:, 0, 0]))
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot Insertion Loss (all networks)
    ax1.plot(freq, il1, label='Network 1')
    ax1.plot(freq, il2, label='Network 2')
    ax1.plot(freq, il_casc, label='Cascaded')
    ax1.plot(freq, il3, label='Network 3', linestyle='--')
    ax1.set_title('Insertion Loss (S21)')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Insertion Loss (dB)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot Return Loss (all networks)
    ax2.plot(freq, rl1, label='Network 1')
    ax2.plot(freq, rl2, label='Network 2')
    ax2.plot(freq, rl_casc, label='Cascaded')
    ax2.plot(freq, rl3, label='Network 3', linestyle='--')
    ax2.set_title('Return Loss (S11)')
    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('Return Loss (dB)')
    ax2.grid(True)
    ax2.legend()
    
    # Plot Comparison: Cascaded vs Network 3
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
    plt.show()'''

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
        ntwk1, ntwk2, cascaded = cascade_s2p(file1_name, file2_name)
        # Read the third S2P file for comparison
        file3_path = os.path.join(os.getcwd(), file3_name)
        ntwk3 = rf.Network(file3_path)
        
        max_eig1, port_powers1 = check_s_parameter_passivity(ntwk1, file1_name)
        plot_passivity(ntwk1.f, max_eig1, port_powers1, file1_name)
        max_eig2, port_powers2 = check_s_parameter_passivity(ntwk2, file2_name)
        plot_passivity(ntwk2.f, max_eig2, port_powers2, file2_name)
        max_eig3, port_powers3 = check_s_parameter_passivity(ntwk3, file3_name)
        plot_passivity(ntwk3.f, max_eig3, port_powers3, file3_name)

        
        
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
