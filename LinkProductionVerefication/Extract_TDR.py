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

def analyze_and_adjust_tdr(time, Z_t, segment_length=0.1E-9, r2_threshold=0.95):
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

def analyze_tdr_segments(time, Z_t, segment_length=0.1E-9, r2_threshold=0.95):
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

        if r_squared > r2_threshold:
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

    # Modify the remaining part of the signal
    Z_t_adj[idx_end:] -= (Tend - Tstart) * slope

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

def main(): 
    Z0 = 50
    parser = argparse.ArgumentParser(description="Process an unknown number of file names.")
    parser.add_argument("files", nargs="+", help="List of file names to process")  # `nargs="+"` means at least one file
    args = parser.parse_args()

    # ========================================================================================================
    # First Code Step is a clean TDR calculation
    # ========================================================================================================
    fig, (ax1, ax2,ax3) = plt.subplots(3, 1, figsize=(10, 12))
    for file in args.files:
        print(f"Processing file: {file}")
        ntwk = rf.Network(file)
        time, s_t = ntwk.s11.step_response(window='hamming')
        Z_t = Z0 * (1 + s_t) / (1 - s_t)
        ax2.plot(time, Z_t, label=file)  # Add label for legend
        time, I_t = ntwk.s11.impulse_response(window='hamming')
        ax1.plot(time, I_t, label=file)  # Add label for legend
        time, s_t = ntwk.s12.step_response(window='hamming')
        ax3.plot(time, s_t, label=file)  # Add label for legend
        
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
        print(f"Processing file: {file}")
        ntwk = rf.Network(file)
        time, s_t = ntwk.s11.step_response(window='hamming')
        Z_t = Z0 * (1 + s_t) / (1 - s_t)
        ax4.plot(time, Z_t, label=file)  # Add label for legend
        
        # Step 1 & 2: Loop through and adjust Z_t for each valid segment
        Z_t_adjusted = analyze_and_adjust_tdr(time, Z_t)
        ax4.plot(time, Z_t_adjusted, label=f"{file} (Adjusted step1&2)", linestyle="--")
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
    ax4.set_xlabel('Time (s)', fontsize=12)
    ax4.set_ylabel('Impedance (Ω)', fontsize=12)
    ax4.grid(True, linestyle='--', alpha=0.7)  # Add grid
    ax4.legend(fontsize=10)  # Add legend
    plt.tight_layout()  # Adjust layout for better spacing

    plt.show()

if __name__ == "__main__":
    main()
