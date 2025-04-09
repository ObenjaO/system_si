import argparse
import skrf as rf
#print(skrf.__version__)
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from s4p_self_cascading import to_db, mixed_mode_s_params

def plot_s_parameters(ntwk1):
    freq = ntwk1.f / 1e9
    
    # Handle both 2-port and 4-port for S21 and S11 (assuming differential or single-ended appropriately)
    if ntwk1.nports == 2:
        il21 = 20 * np.log10(np.abs(ntwk1.s[:, 1, 0]))
        rl11 = 20 * np.log10(np.abs(ntwk1.s[:, 0, 0]))
        il12 = 20 * np.log10(np.abs(ntwk1.s[:, 0, 1]))
        rl22 = 20 * np.log10(np.abs(ntwk1.s[:, 1, 1]))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        ax1.plot(freq, il21, label='S21')
        ax1.plot(freq, il12, label='S12')
        ax1.set_title('Insertion Loss compare')
        ax1.set_xlabel('Frequency (GHz)')
        ax1.set_ylabel('Insertion Loss (dB)')
        ax1.grid(True)
        ax1.legend()
        ax2.plot(freq, rl11, label='S11')
        ax2.plot(freq, rl22, label='S22')
        ax2.set_title('Return Loss compare')
        ax2.set_xlabel('Frequency (GHz)')
        ax2.set_ylabel('Return Loss (dB)')
        ax2.grid(True)
        ax2.legend()
        plt.tight_layout()
    else:  # 4-port, assume differential S21_dd and S11_dd
        mm_ntwk1 = mixed_mode_s_params(ntwk1.s)
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))  # 2x2 grid
        axes = axes.flatten()  # Flatten for easy indexing
        # Plot 1: Differential IL (Sdd21)
        axes[0].plot(freq, to_db(mm_ntwk1['sdd21']), label='Sdd21')
        axes[0].plot(freq, to_db(mm_ntwk1['sdd12']), label='Sdd12')
        axes[0].set_title('Differential IL')
        axes[0].set_ylabel('dB')
        axes[0].grid(True)
        axes[0].legend()

        # Plot 2: Differential RL (Sdd11)
        axes[1].plot(freq, to_db(mm_ntwk1['sdd11']), label='Sdd11')
        axes[1].plot(freq, to_db(mm_ntwk1['sdd22']), label='Sdd22')
        axes[1].set_title('Differential RL ')
        axes[1].set_ylabel('dB')
        axes[1].grid(True)
        axes[1].legend()
        plt.tight_layout()
        
        # Create a new figure for all S-parameters
        fig, axs = plt.subplots(3, 2, figsize=(12, 15))  # 3x2 grid
        axs = axs.flatten()  # Flatten for easy indexing
        
        # Plot all S-parameters (16 combinations for 4-port)
        # We'll plot them in order S11, S12, S13, S14, S21, S22, etc.
        for idx, (i, j) in enumerate([(x, y) for x in range(4) for y in range(x+1,4)]):
            if idx >= len(axs):  # In case we have more S-parameters than subplots
                break
            axs[idx].plot(freq, to_db(ntwk1.s[:, j, i]), label=f"S{j+1}{i+1}")
            axs[idx].plot(freq, to_db(ntwk1.s[:, i, j]), label=f"S{i+1}{j+1}")
            
            axs[idx].set_xlabel('Frequency (GHz)')
            axs[idx].set_ylabel('dB')
            axs[idx].grid(True)
            axs[idx].legend()
    plt.show()

def accumalte_plot_s_parameters(ntwk1, fig=None, axes=None, label=None):
    """
    Plot S-parameters, with option to add to existing plots
    
    Args:
        ntwk1: Network object to plot
        fig: Existing figure object (optional)
        axes: Existing axes array (optional)
        label: Label for this dataset (optional)
        
    Returns:
        fig, axes: Figure and axes objects for further plotting
    """
    freq = ntwk1.f / 1e9
    
    # Create new figure if none provided
    if fig is None or axes is None:
        if ntwk1.nports == 2:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
            axes = [ax1, ax2]
        else:  # 4-port
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Handle both 2-port and 4-port
    if ntwk1.nports == 2:
        il21 = 20 * np.log10(np.abs(ntwk1.s[:, 1, 0]))
        rl11 = 20 * np.log10(np.abs(ntwk1.s[:, 0, 0]))
        il12 = 20 * np.log10(np.abs(ntwk1.s[:, 0, 1]))
        rl22 = 20 * np.log10(np.abs(ntwk1.s[:, 1, 1]))
        
        # Use provided label or default
        label_suffix = f" ({label})" if label else ""
        
        axes[0].plot(freq, il21, label=f'S21{label_suffix}')
        axes[0].plot(freq, il12, label=f'S12{label_suffix}')
        axes[0].set_title('Insertion Loss compare')
        axes[0].set_xlabel('Frequency (GHz)')
        axes[0].set_ylabel('Insertion Loss (dB)')
        axes[0].grid(True)
        axes[0].legend()
        
        axes[1].plot(freq, rl11, label=f'S11{label_suffix}')
        axes[1].plot(freq, rl22, label=f'S22{label_suffix}')
        axes[1].set_title('Return Loss compare')
        axes[1].set_xlabel('Frequency (GHz)')
        axes[1].set_ylabel('Return Loss (dB)')
        axes[1].grid(True)
        axes[1].legend()
        
    else:  # 4-port, assume differential S21_dd and S11_dd
        mm_ntwk1 = mixed_mode_s_params(ntwk1.s)
        label_suffix = f" ({label})" if label else ""
        
        # Differential plots
        axes[0].plot(freq, to_db(mm_ntwk1['sdd21']), label=f'Sdd21{label_suffix}')
        axes[0].plot(freq, to_db(mm_ntwk1['sdd12']), label=f'Sdd12{label_suffix}')
        axes[0].set_title('Differential IL')
        axes[0].set_ylabel('dB')
        axes[0].grid(True)
        axes[0].legend()

        axes[1].plot(freq, to_db(mm_ntwk1['sdd11']), label=f'Sdd11{label_suffix}')
        axes[1].plot(freq, to_db(mm_ntwk1['sdd22']), label=f'Sdd22{label_suffix}')
        axes[1].set_title('Differential RL')
        axes[1].set_ylabel('dB')
        axes[1].grid(True)
        axes[1].legend()
    
    plt.tight_layout()
    return fig, axes


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
    file1_name = ".\Example_S_parameters\QSFP112_Mated_File_HCB-WT27939_MCB-WT27320_RX1_trunc.s4p"
    file2_name = ".\Example_S_parameters\QSFP112_Mated_File_HCB-WT27939_MCB-WT27320_RX2_trunc.s4p"
    file3_name = ".\Example_S_parameters\Diff_5in_85ohm.s4p"
    file4_name = ".\Example_S_parameters\Diff_5in.s4p"
    '''
    file1_name = "diff_cascde_10in_from_1in.s4p"
    file2_name = "diff_cascde_10in_from_5in.s4p"
    #file1_name = ".\Example_S_parameters\Diff_1in.s4p"
    #file2_name = ".\Example_S_parameters\Diff_5in.s4p"
    file3_name = ".\Example_S_parameters\Diff_10in.s4p"
    '''
    ntwk1 = rf.Network(file1_name)
    ntwk2 = rf.Network(file2_name)
    ntwk3 = rf.Network(file1_name)
    ntwk4 = rf.Network(file2_name)
    try:
        plot_parameters(ntwk1, ntwk2, ntwk3, ntwk4)
        plt.show()
        
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
