import argparse
import skrf as rf
#print(skrf.__version__)
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


def plot_parameters(ntwk1, ntwk2,ntwk3, ntwk4 ):
    freq = ntwk1.f / 1e9
    
    # Handle both 2-port and 4-port for S21 and S11 (assuming differential or single-ended appropriately)
    if ntwk1.nports == 2:
        il1 = 20 * np.log10(np.abs(ntwk1.s[:, 1, 0]))
        il2 = 20 * np.log10(np.abs(ntwk2.s[:, 1, 0]))
        il3 = 20 * np.log10(np.abs(ntwk3.s[:, 1, 0]))
        il4 = 20 * np.log10(np.abs(ntwk4.s[:, 1, 0]))
        
        rl1 = 20 * np.log10(np.abs(ntwk1.s[:, 0, 0]))
        rl2 = 20 * np.log10(np.abs(ntwk2.s[:, 0, 0]))
        rl3 = 20 * np.log10(np.abs(ntwk3.s[:, 0, 0]))
        rl4 = 20 * np.log10(np.abs(ntwk4.s[:, 0, 0]))
       
    else:  # 4-port, assume differential S21_dd and S11_dd
        il1 = 20 * np.log10(np.abs((ntwk1.s[:, 1, 0] - ntwk1.s[:, 1, 2] - ntwk1.s[:, 3, 0] + ntwk1.s[:, 3, 2]) / 2))
        il2 = 20 * np.log10(np.abs((ntwk2.s[:, 1, 0] - ntwk2.s[:, 1, 2] - ntwk2.s[:, 3, 0] + ntwk2.s[:, 3, 2]) / 2))
        il3 = 20 * np.log10(np.abs((ntwk3.s[:, 1, 0] - ntwk3.s[:, 1, 2] - ntwk3.s[:, 3, 0] + ntwk3.s[:, 3, 2]) / 2))
        il4 = 20 * np.log10(np.abs((ntwk4.s[:, 1, 0] - ntwk4.s[:, 1, 2] - ntwk4.s[:, 3, 0] + ntwk4.s[:, 3, 2]) / 2))
        
        rl1 = 20 * np.log10(np.abs((ntwk1.s[:, 0, 0] - ntwk1.s[:, 0, 2] - ntwk1.s[:, 2, 0] + ntwk1.s[:, 2, 2]) / 2))
        rl2 = 20 * np.log10(np.abs((ntwk2.s[:, 0, 0] - ntwk2.s[:, 0, 2] - ntwk2.s[:, 2, 0] + ntwk2.s[:, 2, 2]) / 2))
        rl3 = 20 * np.log10(np.abs((ntwk3.s[:, 0, 0] - ntwk3.s[:, 0, 2] - ntwk3.s[:, 2, 0] + ntwk3.s[:, 2, 2]) / 2))
        rl4 = 20 * np.log10(np.abs((ntwk4.s[:, 0, 0] - ntwk4.s[:, 0, 2] - ntwk4.s[:, 2, 0] + ntwk4.s[:, 2, 2]) / 2)) 
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    ax1.plot(freq, il1, label='6 x 2.5inch cascade')
    ax1.plot(freq, il2, label='3 x 5.0inch cascade')
    ax1.plot(freq, il3, label='10inch + 5inch cascade')
    ax1.plot(freq, il4, label='reference 15inch')
    ax1.set_title('Insertion Loss compare')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Insertion Loss (dB)')
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(freq, rl1, label='6 x 2.5inch cascade')
    ax2.plot(freq, rl2, label='3 x 5.0inch cascade')
    ax2.plot(freq, rl3, label='10inch + 5inch cascade')
    ax2.plot(freq, rl4, label='reference 15inch')
    ax2.set_title('Return Loss compare')
    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('Return Loss (dB)')
    ax2.grid(True)
    ax2.legend()
        
    plt.tight_layout()

    fig1, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 12))
    ax3.plot(freq, il4-il1, label='6 x 2.5inch cascade')
    ax3.plot(freq, il4-il2, label='3 x 5.0inch cascade')
    ax3.plot(freq, il4-il3, label='10inch + 5inch cascade')
    ax3.set_title('Insertion Loss error compare to reference')
    ax3.set_xlabel('Frequency (GHz)')
    ax3.set_ylabel('Insertion Loss Error (dB)')
    ax3.grid(True)
    ax3.legend()

    ax4.plot(freq, 100*(il4-il1)/il4, label='6 x 2.5inch cascade')
    ax4.plot(freq, 100*(il4-il2)/il4, label='3 x 5.0inch cascade')
    ax4.plot(freq, 100*(il4-il3)/il4, label='10inch + 5inch cascade')
    ax4.set_title('Insertion Loss error compare to reference')
    ax4.set_xlabel('Frequency (GHz)')
    ax4.set_ylabel('Insertion Loss Error (%)')
    ax4.grid(True)
    ax4.legend()

def main():

    # File names in current directory
    file1_name = 'cascade_2p5in_2p5in_2p5in_2p5in_2p5in_2p5in.s2p'
    file2_name = 'cascade_5in_5in_5in.s2p'
    file3_name = 'cascade_5in_10in.s2p'
    file4_name = '.\Example_S_parameters\Stripline_bend_Strip_bend_cutout_2mm_extended_15n.s2p'
    
    ntwk1 = rf.Network(file1_name)
    ntwk2 = rf.Network(file2_name)
    ntwk3 = rf.Network(file3_name)
    ntwk4 = rf.Network(file4_name)
    try:
        plot_parameters(ntwk1, ntwk2, ntwk3, ntwk4)
        plt.show()
        
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
