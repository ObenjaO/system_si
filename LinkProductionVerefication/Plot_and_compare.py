import argparse
import skrf as rf
#print(skrf.__version__)
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from s4p_self_cascading import to_db, mixed_mode_s_params


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
    else:  # 4-port, assume differential S21_dd and S11_dd
        mm_ntwk1 = mixed_mode_s_params(ntwk1.s)
        mm_ntwk2 = mixed_mode_s_params(ntwk2.s)
        mm_ntwk3 = mixed_mode_s_params(ntwk3.s)
        name1 = "10 x 1[inch]"
        name2 = "2 x 5[inch]"
        name3 = "10[inch]"
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # 2x2 grid
        axes = axes.flatten()  # Flatten for easy indexing

        # Plot 1: Differential IL (Sdd21)
        axes[0].plot(freq, to_db(mm_ntwk1['sdd21']), label=name1)
        axes[0].plot(freq, to_db(mm_ntwk2['sdd21']), label=name2)
        axes[0].plot(freq, to_db(mm_ntwk3['sdd21']), label=name3)
        axes[0].set_title('Differential IL (Sdd21)')
        axes[0].set_ylabel('dB')
        axes[0].grid(True)
        axes[0].legend()

        # Plot 2: Differential RL (Sdd11)
        axes[1].plot(freq, to_db(mm_ntwk1['sdd11']), label=name1)
        axes[1].plot(freq, to_db(mm_ntwk2['sdd11']), label=name2)
        axes[1].plot(freq, to_db(mm_ntwk3['sdd11']), label=name3)
        axes[1].set_title('Differential RL (Sdd11)')
        axes[1].set_ylabel('dB')
        axes[1].grid(True)
        axes[1].legend()

        # Plot 3: Mode Conversion (Scd21)
        axes[2].plot(freq, to_db(mm_ntwk1['scd21']), label=name1)
        axes[2].plot(freq, to_db(mm_ntwk2['scd21']), label=name2)
        axes[2].plot(freq, to_db(mm_ntwk3['scd21']), label=name3)
        axes[2].set_title('Differential→Common IL (Scd21)')
        axes[2].set_ylabel('dB')
        axes[2].grid(True)
        axes[2].legend()

        # Plot 4: Common-mode RL (Scc11)
        axes[3].plot(freq, to_db(mm_ntwk1['scc11']), label=name1)
        axes[3].plot(freq, to_db(mm_ntwk2['scc11']), label=name2)
        axes[3].plot(freq, to_db(mm_ntwk3['scc11']), label=name3)
        axes[3].set_title('Common-mode RL (Scc11)')
        axes[3].set_ylabel('dB')
        axes[3].grid(True)
        axes[3].legend()
    
    plt.tight_layout()
    

def main():

    # File names in current directory
    ''' for old sompare '''
    file1_name = ".\Example_S_parameters\Stripline_2in_44p5_lossy.s2p"
    file2_name = ".\Example_S_parameters\Stripline_2in_51Ohm_lossy.s2p"
    file3_name = ".\Example_S_parameters\Stripline_2in_54p5_lossy.s2p"
    file4_name = ".\Example_S_parameters\Stripline_bend_Strip_bend_cutout_2mm_extended_15n.s2p"
    '''
    file1_name = "diff_cascde_10in_from_1in.s4p"
    file2_name = "diff_cascde_10in_from_5in.s4p"
    #file1_name = ".\Example_S_parameters\Diff_1in.s4p"
    #file2_name = ".\Example_S_parameters\Diff_5in.s4p"
    file3_name = ".\Example_S_parameters\Diff_10in.s4p"
    '''
    ntwk1 = rf.Network(file1_name)
    ntwk2 = rf.Network(file2_name)
    ntwk3 = rf.Network(file3_name)
    ntwk4 = rf.Network(file3_name)
    try:
        plot_parameters(ntwk1, ntwk2, ntwk3, ntwk4)
        plt.show()
        
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
