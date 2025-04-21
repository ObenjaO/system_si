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
from scipy.optimize import curve_fit 
from scipy.fft import fft, ifft,fftfreq

def insertion_loss_test(ntwk, frequency, TH):
    """
    Measure insertion loss of a network at a specific frequency and threshold.
    
    Parameters:
        network (object): Network component/device under test
        frequency (float): Test frequency in Hz
        TH (float): Threshold value for pass/fail determination
        
    Returns:
        tuple: (measured_loss (float), pass_status (bool))
    """
    Freq = ntwk.f
    if ntwk.nports == 4:
        mm_ntwk = mixed_mode_s_params(ntwk.s)
        IL1 = to_db(mm_ntwk['sdd21'])
        IL2 = to_db(mm_ntwk['sdd12'])
    else:
        IL1 = to_db(ntwk.s[:, 1, 0])
        IL2 = to_db(ntwk.s[:, 0, 1])
    # Implementation example - replace with actual measurement logic
    idx_freq_under_test = np.searchsorted(Freq, frequency)
    pass_status1 = IL1[idx_freq_under_test] >= TH
    pass_status2 = IL2[idx_freq_under_test] >= TH
    pass_status = pass_status1 and pass_status2
    return IL1[idx_freq_under_test], IL2[idx_freq_under_test], pass_status

def return_loss_test(ntwk, frequency, TH):
    """
    Measure return loss of a network at a specific frequency and threshold.
    
    Parameters:
        network (object): Network component/device under test
        frequency (float): Test frequency in Hz
        TH (float): Threshold value for pass/fail determination
        
    Returns:
        tuple: (measured_return_loss (float), pass_status (bool))
    """
    Freq = ntwk.f
    if ntwk.nports == 4:
        mm_ntwk = mixed_mode_s_params(ntwk.s)
        RL1 = to_db(mm_ntwk['sdd11'])
        RL2 = to_db(mm_ntwk['sdd22'])
    else:
        RL1 = to_db(ntwk.s[:, 0, 0])
        RL2 = to_db(ntwk.s[:, 1, 1])
    # Implementation example - replace with actual measurement logic
    idx_freq_under_test = np.searchsorted(Freq, frequency)
    pass_status1 = RL1[idx_freq_under_test] <= TH
    pass_status2 = RL2[idx_freq_under_test] <= TH
    pass_status = pass_status1 and pass_status2
    return RL1[idx_freq_under_test], RL2[idx_freq_under_test], pass_status

def diffrential2common_loss_test(ntwk, frequency, TH):
    """
    Measure insertion loss of a network at a specific frequency and threshold.
    
    Parameters:
        network (object): Network component/device under test
        frequency (float): Test frequency in Hz
        TH (float): Threshold value for pass/fail determination
        
    Returns:
        tuple: (measured_loss (float), pass_status (bool))
    """
    Freq = ntwk.f
    if ntwk.nports == 4:
        mm_ntwk = mixed_mode_s_params(ntwk.s)
        d2c1 = to_db(mm_ntwk['sdc21'])
        d2c2 = to_db(mm_ntwk['sdc12'])
    else:
        print("diffrential2common_loss_test: input file must be S4P", file=sys.stderr)
        sys.exit(1)  # Non-zero exit code indicates failure
    
    # Implementation example - replace with actual measurement logic
    idx_freq_under_test = np.searchsorted(Freq, frequency)
    pass_status1 = d2c1[idx_freq_under_test] <= TH
    pass_status2 = d2c2[idx_freq_under_test] <= TH
    pass_status = pass_status1 and pass_status2
    return d2c1[idx_freq_under_test], d2c2[idx_freq_under_test], pass_status

# Define the fitting function
def il_fit_func(f, a0, a1, a2, a4):
    return a0 + a1*np.sqrt(f) + a2*f + a4*(f**2)

def window_function(fn, fb, ft, fr):
    """Compute the window function W(fn) based on the given formula.

    Parameters:
    fn : float or array-like
        The frequency point(s).
    fb : float
        The signaling rate.
    ft : float
        The 3 dB transmit filter bandwidth.
    fr : float
        The 3 dB reference receiver bandwidth.

    Returns:
    float or np.ndarray
        The computed window function value(s).
    """
    sinc_term = np.sinc(fn / fb) ** 2  # sinc(x) in numpy is normalized as sinc(pi*x)/(pi*x)
    tx_filter = (1 / (1 + (fn / ft)) ** 4)
    rx_filter = (1 / (1 + (fn / fr)) ** 8)
    
    return sinc_term * tx_filter * rx_filter 

def compute_fom_ild(fn, ild, fb, ft, fr,debug_mode):
    """Compute the Figure of Merit (FOM_ILD) based on the given formula.

    Parameters:
    fn : array-like
        The frequency points.
    ild : array-like
        The ILD values at each frequency point.
    fb : float
        The signaling rate.
    ft : float
        The 3 dB transmit filter bandwidth.
    fr : float
        The 3 dB reference receiver bandwidth.

    Returns:
    float
        The computed FOM_ILD value.
    """
    # Compute the window function W(fn)
    W_fn = (np.sinc(fn / fb) ** 2) * (1 / (1 + (fn / ft)) ** 4) * (1 / (1 + (fn / fr)) ** 8)
    
    # Compute ILD^2(fn)
    ILD_sq = ild ** 2
    
    # Compute W(fn) * ILD^2(fn)
    W_ILD_sq = W_fn * ILD_sq
    # Compute the FOM_ILD value
    N = len(fn)
    fom_ild = np.sqrt(np.sum(W_fn * (ild ** 2)) / N)
    if (debug_mode):
        # Plot results
        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

        # Plot W(fn)
        axes[0].plot(fn, W_fn, label=r"$W(f_n)$", color='b')
        axes[0].set_ylabel("Window Function")
        axes[0].legend()
        axes[0].grid(True)

        # Plot ILD^2(fn)
        axes[1].plot(fn, ild, label=r"$ILD(f_n)$", color='r')
        axes[1].set_ylabel("ILD")
        axes[1].legend()
        axes[1].grid(True)

        # Plot W(fn) * ILD^2(fn)
        axes[2].plot(fn, W_ILD_sq, label=r"$W(f_n) \cdot ILD^2(f_n)$", color='g')
        axes[2].set_xlabel("Frequency (Hz)")
        axes[2].set_ylabel("Weighted ILD Squared")
        axes[2].legend()
        axes[2].grid(True)

        plt.suptitle(f"FOM_ILD = {fom_ild:.4f}")
        
    return fom_ild

def max_ILD_test(ntwk, frequency_rate, frequency_start, frequency_end, TH_ILD, TH_FOM, debug_mode):
    """
    Measure maximum insertion loss deviation across a frequency range.
    
    Parameters:
        network (object): Network component/device under test
        frequency_start (float): Start frequency in Hz
        frequency_end (float): End frequency in Hz
        TH (float): Threshold value for maximum allowed deviation
        
    Returns:
        tuple: (max_deviation (float), pass_status (bool))
    """
    # calulate the rise and fall time from the Nyqist. The 4.5 can be change to 5 or 4
    Tr = (1/frequency_rate)/4.5
    Tf = (1/frequency_rate)/4.5
    # The 20% to 80% 3dB filter
    ft = 0.2365/Tr
    fr = 0.2365/Tf
    Freq = ntwk.f
    freq_mask = (Freq >= frequency_start) & (Freq <= frequency_end)
    frequencies = ntwk.f[freq_mask]
    if ntwk.nports == 4:
        mm_ntwk = mixed_mode_s_params(ntwk.s)
        sdd21_ntwk = to_db(mm_ntwk['sdd21'])
        #sdd21_ntwk = rf.Network(f=frequencies, s=mm_ntwk['sdd21'][:, np.newaxis, np.newaxis])
        s21_db = sdd21_ntwk[freq_mask]
    else:
        s21_db = to_db(ntwk.s[freq_mask, 0, 0])

    #s21_db =  to_db(S21) 
    # Perform the curve fitting
    try:
        # Initial parameter guesses
        p0 = [np.mean(s21_db), 0, 0, 0]
        plot_results = debug_mode
        # Perform the fit (convert frequencies to GHz for better numerical stability)
        fit_params, _ = curve_fit(il_fit_func, frequencies/1e9, s21_db, p0=p0)
        #print(f"fit_params {fit_params}")
        # Calculate fitted curve
        il_fit = il_fit_func(frequencies/1e9, *fit_params)
        
        # Calculate deviations from fitted curve
        deviations = s21_db - il_fit
        
        # Calculate statistics
        max_deviation = np.max(deviations)
        min_deviation = np.min(deviations)
        avg_deviation = np.mean(np.abs(deviations))
        
        # Pass/fail criteria (worst case deviation must be <= threshold)
        worst_case_deviation = max(abs(max_deviation), abs(min_deviation))
        pass_fail_ILD = worst_case_deviation <= TH_ILD

        fom_ild_value = compute_fom_ild(frequencies, deviations, frequency_rate, ft, fr,debug_mode)
        #print("FOM_ILD:", fom_ild_value)
        pass_fail_FOM = fom_ild_value <= TH_FOM

        if plot_results:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
            ax1.plot(frequencies/1e9, s21_db, 'b-', label='Measured IL')
            ax1.plot(frequencies/1e9, il_fit, 'r--', label='Fitted IL')
            ax1.set_xlabel('Frequency (GHz)')
            ax1.set_ylabel('Insertion Loss (dB)')
            ax1.set_title(f'ILD Analysis ({"PASS" if pass_fail_ILD else "FAIL"})')
            ax1.legend()
            ax2.plot(frequencies/1e9, il_fit - s21_db, 'r--', label='Fitted IL')
            ax2.set_xlabel('Frequency (GHz)')
            ax2.set_ylabel('Insertion Loss Deviation (dB)')
            

            #Perform FFT on ILD_curve
            N = len(frequencies)
            fft_values = fft(il_fit - s21_db)
            fft_frequencies = fftfreq(N, d=(frequencies[1] - frequencies[0]))  # Compute frequency axis
            periods = 1 / np.abs(fft_frequencies)  # Convert frequency to period

            # Take the magnitude of the FFT
            fft_magnitudes = np.abs(fft_values)

            # Apply Inverse FFT to the ILD_curve
            ifft_values = ifft(il_fit - s21_db)

            # Compute frequency corresponding to time axis
            time_axis = np.fft.fftfreq(len(frequencies), d=(frequencies[1] - frequencies[0]))
            # Take the magnitude of the IFFT values
            ifft_magnitudes = np.abs(ifft_values)
            # Only keep the positive side of the IFFT
            positive_indices = time_axis > 0
            time_axis_positive = time_axis[positive_indices]
            ifft_magnitudes_positive = ifft_magnitudes[positive_indices]
            ax3.stem(time_axis_positive, ifft_magnitudes_positive, basefmt=" ")
            ax3.set_xlabel("Time-like Axis (1/Frequency)")
            ax3.set_ylabel("Magnitude (from IFFT)")
            ax3.set_title("IFFT of ILD Curve")
            
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return (max_deviation, min_deviation, avg_deviation, pass_fail_ILD,pass_fail_FOM, 
                frequencies, deviations, fit_params,fom_ild_value)
    
    except RuntimeError:
        raise RuntimeError("Curve fitting failed - check your data and frequency range")


def effective_return_loss(network, frequency, TH):
    """
    Calculate effective return loss accounting for multiple reflections.
    
    Parameters:
        network (object): Network component/device under test
        frequency (float): Test frequency in Hz
        TH (float): Threshold value for pass/fail determination
        
    Returns:
        tuple: (effective_return_loss (float), pass_status (bool))
    """
    # More sophisticated implementation than basic return loss
    s11 = network.measure_s11(frequency)
    s21 = network.measure_s21(frequency)
    erl = 20 * np.log10(abs(s11) + (abs(s21)**2 * abs(s11) / (1 - abs(s11)**2)))
    pass_status = erl >= TH
    return erl, pass_status


# Example usage
if __name__ == "__main__":
    # Load your network data
    #ntwk = rf.Network('.\Example_S_parameters\Diff_5in_85ohm.s4p')
    ntwk = rf.Network('.\Example_S_parameters\QSFP112_Mated_File_HCB-WT27939_MCB-WT27320_RX4.s4p')
    #ntwk = rf.Network('.\cascade_5in_85_100_90_100_85.s4p')
    
    # Define frequency range and threshold
    f_start = 1e6  # 1 GHz
    threshold = 1.0  # 1 dB threshold
    frequency_rate = 26.5626e9
    f_end = frequency_rate    # 6 GHz

    try:
        max_dev, min_dev, avg_dev, passed, passed_FOM, freqs, devs, params, fom_ild_value = max_ILD_test(ntwk, frequency_rate, f_start, f_end, threshold,threshold,debug_mode=1)
        
        print(f"Fitted parameters: a0={params[0]:.3f}, a1={params[1]:.3f}, a2={params[2]:.3f}, a4={params[3]:.3f}")
        print(f"Maximum deviation: {max_dev:.3f} dB")
        print(f"Minimum deviation: {min_dev:.3f} dB")
        print(f"Average absolute deviation: {avg_dev:.3f} dB")
        print(f"FOM ILD: {fom_ild_value:.3f} dB")
        print(f"Passed ILD threshold test: {'Yes' if passed else 'No'}")
        print(f"Passed ILD FOM threshold test: {'Yes' if passed_FOM else 'No'}")
        
        # Optional plotting
       
    except RuntimeError as e:
        print(f"Error: {str(e)}")