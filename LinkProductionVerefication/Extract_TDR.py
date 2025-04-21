import argparse
import skrf as rf
#print(skrf.__version__)
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.stats import norm
import numpy as np
from scipy.stats import linregress
import os
import sys
from s4p_self_cascading import to_db, mixed_mode_s_params

import numpy as np
import scipy.stats as stats
import skrf as rf
import matplotlib.pyplot as plt


def add_dc_frequency_if_missing(ntwk):
    """
    Check if the DC frequency is missing from the S-parameters and add it by extrapolating the first 5 frequencies
    for all ports (e.g., S11, S21, S12, S22 for a 2-port network).

    Parameters:
    ntwk : skrf.Network
        The network object containing S-parameters.

    Returns:
    skrf.Network
        A new network object with the added DC frequency, if necessary.
    """
    # Extract the frequency and S-parameters from the network
    f = ntwk.f
    s_params = ntwk.s

    # Check if DC frequency (0 Hz) is already present in the frequency array
    if 0 not in f:
        # Number of ports
        n_ports = s_params.shape[1]
        n_freqs = f.size

        # Initialize array for new S-parameters (with DC added)
        new_s_params = np.zeros((n_freqs + 1, n_ports, n_ports), dtype=complex)

        # Copy existing S-parameters to the new array (starting at index 1)
        new_s_params[1:, :, :] = s_params

        # Extrapolate DC value for each S-parameter (Sij)
        for i in range(n_ports):  # Input port
            for j in range(n_ports):  # Output port
                # Select the first 5 frequency points and corresponding S-parameters
                first_5_freq = f[:5]
                first_5_s = s_params[:5, i, j]

                # Perform linear regression separately on real and imaginary parts
                slope_real, intercept_real, _, _, _ = stats.linregress(first_5_freq, first_5_s.real)
                slope_imag, intercept_imag, _, _, _ = stats.linregress(first_5_freq, first_5_s.imag)

                # Extrapolate to DC (frequency = 0)
                dc_value = intercept_real + 1j * intercept_imag
                #print(f"DC frequency added to Port{i}{j} - {dc_value}")
                # Assign the extrapolated DC value
                new_s_params[0, i, j] = dc_value

        # Create new frequency array with DC (0 Hz) at the front
        new_frequencies = np.insert(f, 0, 0)

        # Create and return a new network
        new_ntwk = rf.Network(f=new_frequencies, s=new_s_params)
        return new_ntwk
    else:
        # If DC frequency is already present, return the original network
        print("DC frequency already present, no need to add.")
        return ntwk
    
def analyze_and_adjust_tdr(time, Z_t, segment_length=0.05E-9, r2_threshold=0.90):
    """Analyze and adjust TDR in steps, updating Z_t after each valid segment."""
    num_points = len(time)
    idx_start = np.searchsorted(time, 0)  # Start from time >= 0
    
    # Loop through the time segments
    while idx_start < num_points:
        Tstart = time[idx_start]
        Tend = Tstart + segment_length
        idx_end = np.searchsorted(time, Tend)

        if idx_end >= num_points:
            break  # Stop if segment exceeds available data

        time_segment = time[idx_start:idx_end]
        Z_segment = Z_t[idx_start:idx_end]

        # Compute linear fit and R^2
        slope, intercept, r_value, _, _ = linregress(time_segment, Z_segment)
        r_squared = r_value ** 2

        if r_squared > r2_threshold:
            # Adjust Z_t if the segment passes the R^2 threshold
            Z_t_adj = Z_t.copy()
            # Apply slope correction within the segment
            Z_t_adj[idx_start:idx_end] -= (time[idx_start:idx_end] - Tstart) * slope
            # Modify the remaining part of the signal
            Z_t_adj[idx_end:] -= (Tend - Tstart) * slope

            # Update the Z_t with the adjusted version
            Z_t = Z_t_adj

            # Optionally print the result for debugging
            print(f"Adjusted segment: {Tstart:.2e} to {Tend:.2e}, Slope: {slope:.6f}, R^2: {r_squared:.4f}")
        
        # Move to the next segment
        idx_start = idx_end
    
    return Z_t

def analyze_tdr_segments(time, Z_t, segment_length=0.1E-9, r2_threshold=0.9):
    """Identify time segments where the linear fit has R² > threshold."""
    valid_segments = []
    num_points = len(time)
    # Start from time >= 0
    start_idx = np.searchsorted(time, 0)
    idx_start = start_idx  # Set the starting point

    while idx_start < num_points:
        Tstart = time[idx_start]
        Tend = Tstart + segment_length
        idx_end = np.searchsorted(time, Tend)

        if idx_end >= num_points:
            break  # Stop if segment exceeds available data

        time_segment = time[idx_start:idx_end]
        Z_segment = Z_t[idx_start:idx_end]

        slope, intercept, r_value, _, _ = linregress(time_segment, Z_segment)
        r_squared = r_value ** 2

        if r_squared > r2_threshold and slope < 0.75E10 and slope > 0:
            valid_segments.append((Tstart, Tend, slope, r_squared))
            print(f"{Tstart:.2e} : {Tend:.2e} | {slope:.2f} Ω/s | {r_squared:.4f}")
        # Move the start index to the next Tend for the next iteration
        idx_start = idx_end
    
    return valid_segments
'''
def analyze_tdr_segments(time, Z_t, segment_length=1e-9):
    start_time = 0
    end_time = time[-1]
    results = []
    
    while start_time + segment_length <= end_time:
        idx_start = np.searchsorted(time, start_time)
        idx_end = np.searchsorted(time, start_time + segment_length)
        
        time_segment = time[idx_start:idx_end]
        Z_segment = Z_t[idx_start:idx_end]
        
        if len(time_segment) > 1:
            slope, intercept, r_value, _, _ = linregress(time_segment, Z_segment)
            results.append((start_time, start_time + segment_length, slope, r_value**2))
        
        start_time += segment_length
    
    # Print table
    print("Time Segment (s)   | Slope (Ω/s)  | R²")
    print("-------------------|-------------|----")
    for t_start, t_end, slope, r2 in results:
        print(f"{t_start:.2e} : {t_end:.2e} | {slope:.2f} Ω/s | {r2:.4f}")
    
    return results
'''
def adjust_tdr(time, Z_t, Tstart, Tend, slope):
    """Modify Z_t based on linear correction for the given segment."""
    Z_t_adj = Z_t.copy()
    
    idx_start = np.searchsorted(time, Tstart)
    idx_end = np.searchsorted(time, Tend)

    # Apply slope correction within the segment
    Z_t_adj[idx_start:idx_end] -= (time[idx_start:idx_end] - Tstart) * slope

    param = np.log(2*Z_t[idx_start]/(Z_t[idx_start]+Z_t[idx_end]))
    param = param/1.5e8
    param = -param/((Tend - Tstart))
    param = 0
    # Modify the remaining part of the signal
    Z_t_adj[idx_end:] -= (Tend - Tstart) * slope/(1 - param)

    return Z_t_adj

def adjust_tdr_analyze(time, Z_t, Tstart, Tend):
    # Find indices corresponding to Tstart and Tend
    idx_start = np.searchsorted(time, Tstart)
    idx_end = np.searchsorted(time, Tend)
    
    # Extract time and impedance in the range
    time_segment = time[idx_start:idx_end]
    Z_segment = Z_t[idx_start:idx_end]
    
    # Perform linear regression
    #slope, intercept, r_value, _, _ = linregress(time_segment.astype(float), Z_segment.astype(float))

    slope, intercept, r_value, _, _ = linregress(time_segment, Z_segment)
    r_squared = r_value**2
    
    print(f"Average Slope: {slope:.6f} Ω/s")
    print(f"R^2 of Linear Fit: {r_squared:.6f}")
    
    # Create adjusted Z_t
    Z_t_adj = np.copy(Z_t)
    
    # Modify the middle segment using the linear slope
    Z_t_adj[idx_start:idx_end] = Z_t_adj[idx_start:idx_end] - (time[idx_start:idx_end] - Tstart) * slope
    
    # Modify the last segment to be constant
    print(f"Average Slope: {slope:.6f} Ω/s")
    Z_t_adj[idx_end:] = Z_t_adj[idx_end:] - (Tend - Tstart) * slope
    
    return Z_t_adj, slope, r_squared

def s_param_to_Impedance(file):
    S_ntwk = rf.Network(file)
    Z0 = S_ntwk.z0[0, 0]
    print(f"Processing file: {file} - Z0 = {Z0}")
    S_ntwk = add_dc_frequency_if_missing(S_ntwk)
    if S_ntwk.nports == 4:
        factor = 2
        ntwk_dict = mixed_mode_s_params(S_ntwk.s)
        # Convert a specific parameter (e.g., sdd11) into an skrf.Network object
        freqs = S_ntwk.f  # Extract frequency array
        sdd11_ntwk = rf.Network(f=freqs, s=ntwk_dict['sdd11'][:, np.newaxis, np.newaxis])
        sdd22_ntwk = rf.Network(f=freqs, s=ntwk_dict['sdd22'][:, np.newaxis, np.newaxis])
        # Compute step response
        time, s11_t = sdd11_ntwk.step_response(window='hamming')
        time, s22_t = sdd22_ntwk.step_response(window='hamming')
        
    else:
        factor = 1
        ntwk = S_ntwk
        time, s11_t = ntwk.s11.step_response(window='hamming')
        time, s22_t = ntwk.s22.step_response(window='hamming')

    Z11_t = factor * Z0 * (1 + s11_t) / (1 - s11_t)
    Z22_t = factor * Z0 * (1 + s22_t) / (1 - s22_t)
    return (time, Z11_t, Z22_t, Z0)

def adjust_Z(Z_t, time):
    valid_segments = analyze_tdr_segments(time, Z_t)
    for Tstart, Tend, slope, r_squared in valid_segments:
        Z_t = adjust_tdr(time, Z_t, Tstart, Tend, slope)
    return Z_t

def main(): 
    parser = argparse.ArgumentParser(description="Process an unknown number of file names.")
    parser.add_argument("files", nargs="+", help="List of file names to process")  # `nargs="+"` means at least one file
    args = parser.parse_args()

    # ========================================================================================================
    # First Code Step is a clean TDR calculation
    # ========================================================================================================
    fig, (ax1, ax2,ax3) = plt.subplots(3, 1, figsize=(10, 12))
    for file in args.files:
        S_ntwk = rf.Network(file)
        Z0 = S_ntwk.z0[0, 0]
        print(f"Processing file: {file} - Z0 = {Z0}")
        S_ntwk = add_dc_frequency_if_missing(S_ntwk)
        if S_ntwk.nports == 4:
            factor = 2
            ntwk_dict = mixed_mode_s_params(S_ntwk.s)
            # Convert a specific parameter (e.g., sdd11) into an skrf.Network object
            freqs = S_ntwk.f  # Extract frequency array
            sdd11_ntwk = rf.Network(f=freqs, s=ntwk_dict['sdd11'][:, np.newaxis, np.newaxis])
            sdd12_ntwk = rf.Network(f=freqs, s=ntwk_dict['sdd21'][:, np.newaxis, np.newaxis])
            # Compute step response
            time, s_t = sdd11_ntwk.step_response(window='hamming')
            time1, I_t = sdd11_ntwk.impulse_response(window='hamming')
            time, step_t = sdd12_ntwk.step_response(window='hamming')
        else:
            factor = 1
            ntwk = S_ntwk
            time, s_t = ntwk.s11.step_response(window='hamming')
            time1, I_t = ntwk.s11.impulse_response(window='hamming')
            time, step_t = ntwk.s12.step_response(window='hamming')
        Z_t = factor * Z0 * (1 + s_t) / (1 - s_t)
        ax2.plot(time, Z_t, label=file)  # Add label for legend
        ax1.plot(time1, I_t, label=file)  # Add label for legend
        ax3.plot(time, step_t, label=file)  # Add label for legend
        
    # Beautify plot
    ax3.set_title('Step transfer', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Amplitude', fontsize=12)
    ax2.set_title('TDR Impedance', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Impedance (Ω)', fontsize=12)
    ax1.set_title('Impulse responde', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.7)  # Add grid
    ax2.grid(True, linestyle='--', alpha=0.7)  # Add grid
    ax1.grid(True, linestyle='--', alpha=0.7)  # Add grid
    ax2.legend(title="Files", fontsize=10)  # Add legend
    plt.tight_layout()  # Adjust layout for better spacing
    # ========================================================================================================
    # Second Code Step is an adjust TDR calculation
    # # ========================================================================================================
    fig1, (ax4) = plt.subplots(1, 1, figsize=(10, 12))

    Tstart = 5E-11
    Tend = 5.1E-9


    for file in args.files:
        S_ntwk = rf.Network(file)
        Z0 = S_ntwk.z0[0, 0]
        print(f"Processing file: {file} - Z0 = {Z0}")
        S_ntwk = add_dc_frequency_if_missing(S_ntwk)
        if S_ntwk.nports == 4:
            factor = 2
            ntwk_dict = mixed_mode_s_params(S_ntwk.s)
            # Convert a specific parameter (e.g., sdd11) into an skrf.Network object
            freqs = S_ntwk.f  # Extract frequency array
            sdd11_ntwk = rf.Network(f=freqs, s=ntwk_dict['sdd11'][:, np.newaxis, np.newaxis])
            # Compute step response
            time, s_t = sdd11_ntwk.step_response(window='hamming')
            
        else:
            factor = 1
            ntwk = S_ntwk
            time, s_t = ntwk.s11.step_response(window='hamming')
        Z_t = factor * Z0 * (1 + s_t) / (1 - s_t)
        ax4.plot(time, Z_t, label=file)  # Add label for legend
        # Step 1 & 2: Loop through and adjust Z_t for each valid segment
        #Z_t_adjusted = analyze_and_adjust_tdr(time, Z_t)
        #ax4.plot(time, Z_t_adjusted, label=f"{file} (Adjusted step1&2)", linestyle="--")
        # Step 1: Identify valid segments
        valid_segments = analyze_tdr_segments(time, Z_t)
        # Step 2 & 3: Adjust TDR for each valid segment
        for Tstart, Tend, slope, r_squared in valid_segments:
            Z_t = adjust_tdr(time, Z_t, Tstart, Tend, slope)
        #analyze_tdr_segments(time, Z_t, segment_length=1e-9)
        #Z_t_adj, slope, r_squared = adjust_tdr(time, Z_t, Tstart, Tend)
        ax4.plot(time, Z_t, label=f"{file} (Adjusted)", linestyle="--")
        

    # Beautify plot
    ax4.set_title('TDR Impedance - Modify S-parameters at DC', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Time (s)', fontsize=16)
    ax4.set_ylabel('Impedance (Ω)', fontsize=16)
    ax4.tick_params(axis='both', labelsize=14) 
    ax4.grid(True, linestyle='--', alpha=0.7)  # Add grid
    ax4.legend(fontsize=10)  # Add legend
    plt.tight_layout()  # Adjust layout for better spacing

    plt.show()

if __name__ == "__main__":
    main()
