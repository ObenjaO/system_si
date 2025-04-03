import argparse
import skrf as rf
#print(skrf.__version__)
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def mixed_mode_s_params(s):
    """Compute mixed-mode S-params with shape validation."""
    if s.ndim != 3 or s.shape[1:] != (4, 4):
        raise ValueError(f"Input must be (freqs, 4, 4). Got {s.shape}")
    # Proceed with calculations...
    # Differential (dd)
    sdd11 = (s[:, 0, 0] - s[:, 0, 1] - s[:, 1, 0] + s[:, 1, 1]) / 2
    sdd21 = (s[:, 2, 0] - s[:, 2, 1] - s[:, 3, 0] + s[:, 3, 1]) / 2
    sdd12 = (s[:, 0, 2] - s[:, 0, 3] - s[:, 1, 2] + s[:, 1, 3]) / 2
    sdd22 = (s[:, 2, 2] - s[:, 2, 3] - s[:, 3, 2] + s[:, 3, 3]) / 2
    
    # Common-mode (cc)
    scc11 = (s[:, 0, 0] + s[:, 0, 1] + s[:, 1, 0] + s[:, 1, 1]) / 2
    scc21 = (s[:, 2, 0] + s[:, 2, 1] + s[:, 3, 0] + s[:, 3, 1]) / 2
    scc12 = (s[:, 0, 2] + s[:, 0, 3] + s[:, 1, 2] + s[:, 1, 3]) / 2
    scc22 = (s[:, 2, 2] + s[:, 2, 3] + s[:, 3, 2] + s[:, 3, 3]) / 2
    
    # Mode conversion (cd/dc)
    sdc11 = (s[:, 0, 0] - s[:, 0, 1] + s[:, 1, 0] - s[:, 1, 1]) / 2  # Common → Differential reflect
    scd11 = (s[:, 0, 0] + s[:, 0, 1] - s[:, 1, 0] - s[:, 1, 1]) / 2  # Differential → Common reflect
    sdc21 = (s[:, 2, 0] - s[:, 2, 1] + s[:, 3, 0] - s[:, 3, 1]) / 2  # Common → Differential transmit
    scd21 = (s[:, 2, 0] + s[:, 2, 1] - s[:, 3, 0] - s[:, 3, 1]) / 2  # Differential → Common transmit
    sdc12 = (s[:, 0, 2] - s[:, 0, 3] + s[:, 1, 2] - s[:, 1, 3]) / 2  # Common → Differential
    scd12 = (s[:, 0, 2] + s[:, 0, 3] - s[:, 1, 2] - s[:, 1, 3]) / 2  # Differential → Common
    sdc22 = (s[:, 2, 2] - s[:, 2, 3] + s[:, 3, 2] - s[:, 3, 3]) / 2  # Common → Differential
    scd22 = (s[:, 2, 2] + s[:, 2, 3] - s[:, 3, 2] - s[:, 3, 3]) / 2  # Differential → Common

    return {
        'sdd11': sdd11, 'sdd21': sdd21, 'sdd12': sdd12, 'sdd22': sdd22,
        'scc11': scc11, 'scc21': scc21, 'scc12': scc12, 'scc22': scc22,
        'sdc11': sdc11, 'scd11': scd11, 'sdc21': sdc21, 'scd21': scd21,
        'sdc12': sdc12, 'scd12': scd12, 'sdc22': sdc22, 'scd22': scd22
    }


def cascade_s2p_or_s4p(file1_name, file2_name, output_name='cascaded'):
    file1_path = os.path.join(os.getcwd(), file1_name)
    file2_path = os.path.join(os.getcwd(), file2_name)
    output_path = os.path.join(os.getcwd(), output_name)
    
    ntwk1 = rf.Network(file1_path)
    ntwk2 = rf.Network(file2_path)
        
    
    if not np.allclose(ntwk1.f, ntwk2.f):
        raise ValueError("Frequency points of the two networks do not match!")
    
    if ntwk1.nports != ntwk2.nports:
        raise ValueError("Both networks must have the same number of ports!")
    
    # Cascade based on number of ports
    if ntwk1.nports == 2:
        cascaded = rf.connect(ntwk1, 1, ntwk2, 0)  # S2P: Port 2 to Port 1
        output_path += '.s2p'
    elif ntwk1.nports == 4:
        cascaded = rf.cascade(ntwk1, ntwk2)
        #cascaded = rf.connect(ntwk1, 3, ntwk2, 0)  # S4P: Port 4 to Port 1
        #cascaded = rf.connect(cascaded, 2, ntwk2, 1)  # Port 3 to Port 2
        output_path += '.s4p'  # Result is 2-port
    else:
        raise ValueError("Only 2-port (S2P) or 4-port (S4P) networks are supported for cascading!")
    
    cascaded.write_touchstone(output_path)
    return ntwk1, ntwk2, cascaded

# Convert to dB (with small offset to avoid log(0))
def to_db(s_param):
    return 20 * np.log10(np.abs(s_param) + 1e-40)

def plot_parameters(ntwk1, ntwk2, cascaded):
    freq = ntwk1.f / 1e9
    
    # Handle both 2-port and 4-port for S21 and S11 (assuming differential or single-ended appropriately)
    if ntwk1.nports == 2:
        il1 = 20 * np.log10(np.abs(ntwk1.s[:, 1, 0]) + 1e-40)
        il2 = 20 * np.log10(np.abs(ntwk2.s[:, 1, 0]) + 1e-40)
        il_casc = 20 * np.log10(np.abs(cascaded.s[:, 1, 0]) + 1e-40)
        
        
        rl1 = 20 * np.log10(np.abs(ntwk1.s[:, 0, 0]) + 1e-40)
        rl2 = 20 * np.log10(np.abs(ntwk2.s[:, 0, 0]) + 1e-40)
        rl_casc = 20 * np.log10(np.abs(cascaded.s[:, 0, 0]) + 1e-40)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
        ax1.plot(freq, il1, label='Network 1')
        ax1.plot(freq, il2, label='Network 2')
        ax1.plot(freq, il_casc, label='Cascaded')
        ax1.set_title('Insertion Loss (S21 or S21_dd)')
        ax1.set_xlabel('Frequency (GHz)')
        ax1.set_ylabel('Insertion Loss (dB)')
        ax1.grid(True)
        ax1.legend()
        
        ax2.plot(freq, rl1, label='Network 1')
        ax2.plot(freq, rl2, label='Network 2')
        ax2.plot(freq, rl_casc, label='Cascaded')
        ax2.set_title('Return Loss (S11 or S11_dd)')
        ax2.set_xlabel('Frequency (GHz)')
        ax2.set_ylabel('Return Loss (dB)')
        ax2.grid(True)
        ax2.legend()
       
    else:  # 4-port, assume differential S21_dd and S11_dd
        mm_ntwk1 = mixed_mode_s_params(ntwk1.s)
        mm_ntwk2 = mixed_mode_s_params(ntwk2.s)
        mm_cascaded = mixed_mode_s_params(cascaded.s)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # 2x2 grid
        axes = axes.flatten()  # Flatten for easy indexing

        # Plot 1: Differential IL (Sdd21)
        axes[0].plot(freq, to_db(mm_ntwk1['sdd21']), label='Network 1')
        axes[0].plot(freq, to_db(mm_ntwk2['sdd21']), label='Network 2')
        axes[0].plot(freq, to_db(mm_cascaded['sdd21']), label='Cascaded')
        axes[0].set_title('Differential IL (Sdd21)')
        axes[0].set_ylabel('dB')
        axes[0].grid(True)
        axes[0].legend()

        # Plot 2: Differential RL (Sdd11)
        axes[1].plot(freq, to_db(mm_ntwk1['sdd11']), label='Network 1')
        axes[1].plot(freq, to_db(mm_ntwk2['sdd11']), label='Network 2')
        axes[1].plot(freq, to_db(mm_cascaded['sdd11']), label='Cascaded')
        axes[1].set_title('Differential RL (Sdd11)')
        axes[1].set_ylabel('dB')
        axes[1].grid(True)
        axes[1].legend()

        # Plot 3: Mode Conversion (Scd21)
        axes[2].plot(freq, to_db(mm_ntwk1['scd21']), label='Network 1')
        axes[2].plot(freq, to_db(mm_ntwk2['scd21']), label='Network 2')
        axes[2].plot(freq, to_db(mm_cascaded['scd21']), label='Cascaded')
        axes[2].set_title('Differential→Common IL (Scd21)')
        axes[2].set_ylabel('dB')
        axes[2].grid(True)
        axes[2].legend()

        # Plot 4: Common-mode RL (Scc11)
        axes[3].plot(freq, to_db(mm_ntwk1['scc11']), label='Network 1')
        axes[3].plot(freq, to_db(mm_ntwk2['scc11']), label='Network 2')
        axes[3].plot(freq, to_db(mm_cascaded['scc11']), label='Cascaded')
        axes[3].set_title('Common-mode RL (Scc11)')
        axes[3].set_ylabel('dB')
        axes[3].grid(True)
        axes[3].legend()
            
    plt.tight_layout()

def main():

    parser = argparse.ArgumentParser(description="Test S parameter passivity, reciprocity & causality.")
    
    # Adding arguments for file path and debug mode
    parser.add_argument('Sfile1_name', type=str, help="The full path to the S-parameter file.")
    parser.add_argument('repeat_number', type=int, help="The number of time to cascade.")
    parser.add_argument('Scascade_name', type=str, help="The name for the cascade S-params.")
    parser.add_argument('debug', type=bool, help="Enable or disable debug mode (True/False).")
    
    # Parsing the command-line arguments
    args = parser.parse_args()
    file1_name = args.Sfile1_name  # This remains correct
    repeat_number = args.repeat_number  # This is now an integer
    cascade_name = args.Scascade_name 
    debug_mode = args.debug
    
    
    
    try:
        if repeat_number < 1:
            raise ValueError("Number of cascades must be greater than 0!")
        # Cascade the networks
        ntwk1, ntwk2, cascaded = cascade_s2p_or_s4p(file1_name, file1_name, output_name=cascade_name)
        # Read the third SnP file for comparison
        suffix = ".s4p"
        if ntwk1.nports == 2:
            suffix = ".s2p"
        for i in range(repeat_number-2):
            print("Start Cascading loop {} ".format(i+1))
            ntwk1, ntwk2, cascaded = cascade_s2p_or_s4p(file1_name, cascade_name + suffix, output_name=cascade_name)
            print("Cascading loop {} completed successfully!".format(i+1))
        print("Cascading completed successfully!")
        # Plot parameters
        if (debug_mode):
            print("Starting plotting!")
            plot_parameters(ntwk1, ntwk2, cascaded)
            plt.show()
        
        print(f"Cascaded S-parameters saved to 'cascaded.s2p' in {os.getcwd()}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
