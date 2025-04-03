import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import ifft, fftshift
import argparse
import skrf as rf
from scipy.signal import convolve
from Extract_TDR import add_dc_frequency_if_missing
from s4p_self_cascading import mixed_mode_s_params

def verify_equalization(frequencies, S21_mirrored, h_time, name):
    # Frequency-domain verification
    H_fir = np.fft.fft(h_time, n=len(frequencies))
    equalized_freq = S21_mirrored * H_fir

    #print(f"Length check: f_extended={len(frequencies)}, S21_mirrored={len(S21_mirrored)}, H_fir={len(H_fir)}")
    #print(f"Max |equalized_freq|={np.max(np.abs(equalized_freq))}, Min |equalized_freq|={np.min(np.abs(equalized_freq))}")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 16))
    # Plot 1: Frequency Response
    ax1.plot(frequencies, 20 * np.log10(np.abs(S21_mirrored)), label="S21_mirrored (dB)")
    ax1.plot(frequencies, 20 * np.log10(np.abs(equalized_freq)), label="Equalized (S21_mirrored * H_fir) (dB)")
    ax1.axhline(0, color='gray', linestyle='--', label="Ideal (0 dB)")
    ax1.set_title(f"Frequency Response: Original vs. Equalized ({name})")  # Fixed: Use set_title()
    ax1.set_xlabel("Frequency (Hz)", fontsize=16)  # Fixed: Use set_xlabel()
    ax1.set_ylabel("Magnitude (dB)", fontsize=16)  # Fixed: Use set_ylabel()
    ax1.tick_params(axis='both', labelsize=14)
    ax1.legend()
    ax1.grid(True)  # Fixed: Apply grid to ax1, not plt

    ax2.plot(frequencies, 20 * np.log10(np.abs(H_fir)), label="H_fir (dB)")
    ax2.set_title("Frequency Response of H_fir")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.legend()
    plt.tight_layout()  # Prevent overlapping labels
    

def compute_tx_fir(frequencies, S21, num_taps=7, apply_window=False):
    H_inv = 1 / (S21 + np.min(abs(S21))/100)
    #print(f"Min |S21|: {np.min(np.abs(S21))}")

    H_inv_mirrored = np.concatenate([np.conj(H_inv[-1:0:-1]), H_inv])
    f_extended = np.concatenate([-frequencies[-1:0:-1], frequencies ])
    
    # Pre-IFFT check
    S21_mirrored = np.concatenate([np.conj(S21[-1:0:-1]), S21])
    
    h_time = fftshift(ifft(H_inv_mirrored))
    #print(f"h_time length {len(h_time)} | H_inv_mirrored {len(H_inv_mirrored)} | S21_mirrored {len(S21_mirrored)} sum h_time {np.sum(h_time)}")
    verify_equalization(f_extended, S21_mirrored, h_time, "Equalized Plot for validation the S parameter")

    # Extract FIR taps
    main_tap_index = np.argmax(np.abs(h_time))
    half_taps = num_taps // 2
    start_idx = max(0, main_tap_index - half_taps)
    end_idx = min(len(h_time), start_idx + num_taps)
    fir_taps = h_time[start_idx:end_idx]

    if apply_window:
        window = np.hamming(len(fir_taps))
        fir_taps *= window

    return fir_taps
    plt.figure(figsize=(10, 4))
    plt.stem(fir_taps)
    plt.title("Tx FIR Tap Coefficients")
    plt.xlabel("Tap Index", fontsize=16)
    plt.ylabel("Coefficient Value", fontsize=16)
    plt.tick_params(axis='both', labelsize=14)
    plt.grid()
    
    fir_taps_padded = np.pad(fir_taps, (0, len(S21_mirrored) - len(fir_taps)), mode='constant')  # Pad to match 
    print(f"sum fir_taps_padded {np.sum(fir_taps_padded)}")
    H_fir_taps = np.fft.fft(fir_taps_padded, n=len(f_extended))
    verify_equalization(f_extended, S21_mirrored, fftshift(fir_taps_padded), "Full fir_taps")
    
    fir_taps_pre = h_time[start_idx+1:end_idx]
    half_taps = num_taps // 2
    fir_taps_pre[half_taps:num_taps] = 0
    if apply_window:
        window = np.hamming(len(fir_taps_pre))
        fir_taps_pre *= window
    fir_taps_padded_pre = np.pad(fir_taps, (0, len(S21_mirrored) - len(fir_taps)), mode='constant')  # Pad to match 
    verify_equalization(f_extended, S21_mirrored, fftshift(fir_taps_padded_pre), "pre course only fir_taps")
    plt.show()
    return fir_taps


def adjust_FIR(frequencies, S21, Tx_FIR,mode ='De-emphasis',adjust_type='pre_only',normalize=True):
    '''
    mode = De-emphasis or Pre-emphasis 
    adjust_type = pre_only or one_post or full
    '''
    f_extended = np.concatenate([-frequencies[-1:0:-1], frequencies ])
    # Pre-IFFT check
    S21_mirrored = np.concatenate([np.conj(S21[-1:0:-1]), S21])
    N = len(Tx_FIR)
    main_index=N//2
    TX_FIR_abs = abs(Tx_FIR)
    sum_tap = sum(TX_FIR_abs)
    if mode.lower() == 'de-emphasis': #reduce the signal during steady states, effectively attenuating low frequencies.
        fir_taps = TX_FIR_abs
        fir_taps[0:N-1:2] = -TX_FIR_abs[0:N-1:2]
    else: #mode = 'Pre-emphasis : boost transitions, and the main tap might stay closer to its nominal value.
        fir_taps = TX_FIR_abs
    if adjust_type=='pre_only':
        fir_taps[(main_index+1):N-1] = 0
    elif adjust_type=='one_post':
        fir_taps[(main_index+2):N-1] = 0
    
    if normalize:
        fir_taps = fir_taps/sum_tap
    fir_taps_padded = np.pad(fir_taps, (0, len(S21_mirrored) - len(fir_taps)), mode='constant')  # Pad to match
    name = mode + ":" + adjust_type
    verify_equalization(f_extended, S21_mirrored, fftshift(fir_taps_padded), name)
    return fir_taps
    

def main():
    parser = argparse.ArgumentParser(description="Process S-parameter files.")
    parser.add_argument("files", nargs="+", help="List of file names")
    args = parser.parse_args()

    for file in args.files:
        print(f"Processing file: {file}")
        try:
            S_ntwk = rf.Network(file)
            S_ntwk = add_dc_frequency_if_missing(S_ntwk)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue

        freqs = S_ntwk.f
        if S_ntwk.nports == 4:
            ntwk_dict = mixed_mode_s_params(S_ntwk.s)
            sdd12_ntwk = rf.Network(f=freqs, s=ntwk_dict['sdd21'][:, np.newaxis, np.newaxis])
        elif S_ntwk.nports == 2:
            sdd12_ntwk = S_ntwk  # Directly use the network object
        else:
            print(f"Unsupported port count: {S_ntwk.nports}")
            continue

        fir_taps = compute_tx_fir(freqs, sdd12_ntwk.s[:, 0, 0], num_taps=7, apply_window=True)
        #Pre-emphasis ; De-emphasis
        final_taps = adjust_FIR(freqs, sdd12_ntwk.s[:, 0, 0], fir_taps,mode ='Pre-emphasis',adjust_type='pre_only',normalize=True)
        print("FIR Taps:", final_taps)
        
        plt.show()

    

if __name__ == "__main__":
    main()