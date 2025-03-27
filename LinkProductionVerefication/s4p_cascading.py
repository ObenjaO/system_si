import argparse
import skrf as rf
#print(skrf.__version__)
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from s4p_self_cascading import cascade_s2p_or_s4p, plot_parameters,to_db, mixed_mode_s_params

'''
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
        cascaded = rf.connect(ntwk1, 3, ntwk2, 0)  # S4P: Port 4 to Port 1
        cascaded = rf.connect(cascaded, 2, ntwk2, 1)  # Port 3 to Port 2
        output_path += '.s2p'  # Result is 2-port
    else:
        raise ValueError("Only 2-port (S2P) or 4-port (S4P) networks are supported for cascading!")
    
    cascaded.write_touchstone(output_path)
    return ntwk1, ntwk2, cascaded


def plot_parameters(ntwk1, ntwk2, cascaded):
    freq = ntwk1.f / 1e9
    
    # Handle both 2-port and 4-port for S21 and S11 (assuming differential or single-ended appropriately)
    if ntwk1.nports == 2:
        il1 = 20 * np.log10(np.abs(ntwk1.s[:, 1, 0]))
        il2 = 20 * np.log10(np.abs(ntwk2.s[:, 1, 0]))
        il_casc = 20 * np.log10(np.abs(cascaded.s[:, 1, 0]))
        
        
        rl1 = 20 * np.log10(np.abs(ntwk1.s[:, 0, 0]))
        rl2 = 20 * np.log10(np.abs(ntwk2.s[:, 0, 0]))
        rl_casc = 20 * np.log10(np.abs(cascaded.s[:, 0, 0]))
       
    else:  # 4-port, assume differential S21_dd and S11_dd
        il1 = 20 * np.log10(np.abs((ntwk1.s[:, 1, 0] - ntwk1.s[:, 1, 2] - ntwk1.s[:, 3, 0] + ntwk1.s[:, 3, 2]) / 2))
        il2 = 20 * np.log10(np.abs((ntwk2.s[:, 1, 0] - ntwk2.s[:, 1, 2] - ntwk2.s[:, 3, 0] + ntwk2.s[:, 3, 2]) / 2))
        il_casc = 20 * np.log10(np.abs(cascaded.s[:, 1, 0]))  # Cascaded is 2-port
        
        
        rl1 = 20 * np.log10(np.abs((ntwk1.s[:, 0, 0] - ntwk1.s[:, 0, 2] - ntwk1.s[:, 2, 0] + ntwk1.s[:, 2, 2]) / 2))
        rl2 = 20 * np.log10(np.abs((ntwk2.s[:, 0, 0] - ntwk2.s[:, 0, 2] - ntwk2.s[:, 2, 0] + ntwk2.s[:, 2, 2]) / 2))
        rl_casc = 20 * np.log10(np.abs(cascaded.s[:, 0, 0]))
        
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
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
        
    plt.tight_layout()

'''
def main():

    parser = argparse.ArgumentParser(description="Test S parameter passivity, reciprocity & causality.")
    
    # Adding arguments for file path and debug mode
    parser.add_argument('Sfile1_name', type=str, help="The full path to the first S-parameter file.")
    parser.add_argument('Sfile2_name', type=str, help="The full path to the first S-parameter file.")
    parser.add_argument('Scascade_name', type=str, help="The name for the cascade S-params.")
    parser.add_argument('debug', type=bool, help="Enable or disable debug mode (True/False).")
    
    # Parsing the command-line arguments
    args = parser.parse_args()
    file1_name = sys.argv[1]
    file2_name = sys.argv[2]
    cascade_name = sys.argv[3]
    debug_mode = sys.argv[4]
        
    try:
        # Cascade the networks
        ntwk1, ntwk2, cascaded = cascade_s2p_or_s4p(file1_name, file2_name, output_name=cascade_name)
        # Read the third SnP file for comparison
        
        # Plot parameters
        if (debug_mode):
            plot_parameters(ntwk1, ntwk2, cascaded)
            plt.show()
        print("Cascading completed successfully!")
        print(f"Cascaded S-parameters saved to 'cascaded.s2p' in {os.getcwd()}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
