# analyze_database.py
import os
import re
import sqlite3
import csv
import pandas as pd
import matplotlib.pyplot as plt

# analyze_returnloss.py
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Dict, Tuple
from collections import Counter, defaultdict

def plot_test_type_histograms(test_type, db_file_path='cascade_analysis.db', show_plots=True):
    # Connect to the database
    conn = sqlite3.connect(db_file_path)

    # Load data for the specified test type into a pandas DataFrame
    # Load data for the specified test type into a pandas DataFrame
    query = f"""
        SELECT metric_1_value, metric_2_value 
        FROM test_results 
        WHERE test_type = '{test_type}'
    """
    test_df = pd.read_sql_query(query, conn)
    

    # Close the connection
    conn.close()

    # Check if there's data to plot
    if test_df.empty:
        print(f"No {test_type} data found in the database.")
        return

    # Define title mappings for specific test types
    title_map = {
        "InsertionLoss": ("IL1 (Insertion Loss 1)", "IL2 (Insertion Loss 2)"),
        "ReturnLoss": ("RL1 (Return Loss 1)", "RL2 (Return Loss 2)"),
        "D2C": ("D2C1 (Differential to Common 1)", "D2C2 (Differential to Common 2)")
    }
    # Default to generic titles if test_type isn't in the map
    metric1_title, metric2_title = title_map.get(test_type, (f"{test_type} Metric 1", f"{test_type} Metric 2"))

    # Create a 2x1 plot (stacked vertically)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f"{test_type} Histograms", fontsize=16)
    axes = axes.flatten()  # Flatten for easier indexing

    # Plot metric_1_value with custom title
    axes[0].hist(test_df['metric_1_value'], bins=25, color='skyblue', edgecolor='black')
    axes[0].set_title(metric1_title)
    axes[0].set_xlabel("dB")
    axes[0].set_ylabel("Frequency")

    # Plot metric_2_value with custom title
    axes[1].hist(test_df['metric_2_value'], bins=25, color='lightgreen', edgecolor='black')
    axes[1].set_title(metric2_title)
    axes[1].set_xlabel("dB")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the suptitle

    # Show the plot if requested
    if show_plots:
        plt.show()

def plot_returnloss_histograms(db_file_path='cascade_analysis.db', show_plots=True):
    # Connect to the database
    conn = sqlite3.connect(db_file_path)

    # Load ReturnLoss data into a pandas DataFrame
    returnloss_df = pd.read_sql_query("""
        SELECT metric_1_value, metric_2_value 
        FROM test_results 
        WHERE test_type = 'ReturnLoss'
    """, conn)

    # Close the connection
    conn.close()

    # Check if there's data to plot
    if returnloss_df.empty:
        print("No ReturnLoss data found in the database.")
        return

    # Create a 2x2 plot (or adjust to 1x2 if preferred)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle("ReturnLoss Histograms", fontsize=16)
    axes = axes.flatten()  # Flatten for easier indexing

    # Plot RL1 (metric_1_value)
    axes[0].hist(returnloss_df['metric_1_value'], bins=25, color='skyblue', edgecolor='black')
    axes[0].set_title("RL1 (Return Loss 1)")
    axes[0].set_xlabel("dB")
    axes[0].set_ylabel("Frequency")

    # Plot RL2 (metric_2_value)
    axes[1].hist(returnloss_df['metric_2_value'], bins=25, color='lightgreen', edgecolor='black')
    axes[1].set_title("RL2 (Return Loss 2)")
    axes[1].set_xlabel("dB")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the suptitle

    # Show the plot if requested
    if show_plots:
        plt.show()

def plot_max_ild_histograms(db_file_path='cascade_analysis.db', show_plots=True):
    # Connect to the database
    conn = sqlite3.connect(db_file_path)

    # Load Max_ILD data into a pandas DataFrame
    max_ild_df = pd.read_sql_query("""
        SELECT metric_1_value, metric_2_value 
        FROM test_results 
        WHERE test_type = 'Max_ILD'
    """, conn)

    # Close the connection
    conn.close()

    # Check if there's data to plot
    if max_ild_df.empty:
        print("No Max_ILD data found in the database.")
        return

    # Create a 2x1 plot (stacked vertically)
    # Create a 2x2 plot (or adjust to 1x2 if preferred)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle("ILD Histograms", fontsize=16)
    axes = axes.flatten()  # Flatten for easier indexing
    
    # Plot RL1 (metric_1_value)
    axes[0].hist(max_ild_df['metric_1_value'], bins=25, color='skyblue', edgecolor='black')
    axes[0].set_title("Max ILD")
    axes[0].set_xlabel("dB")
    axes[0].set_ylabel("Frequency")

    # Plot RL2 (metric_2_value)
    axes[1].hist(max_ild_df['metric_2_value'], bins=25, color='lightgreen', edgecolor='black')
    axes[1].set_title("FOM ILD")
    axes[1].set_xlabel("dB")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the suptitle

    # Show the plot if requested
    if show_plots:
        plt.show()

# Function to create a 2x2 histogram plot
def plot_2x2_histograms(data, param_list, title_prefix, value_col, fig_num):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"{title_prefix} Histograms - Figure {fig_num}", fontsize=16)
    axes = axes.flatten()  # Flatten the 2x2 grid for easier iteration

    for i, param in enumerate(param_list[:4]):  # Take up to 4 parameters per figure
        param_data = data[data[value_col[0]] == param][value_col[1]]
        axes[i].hist(param_data, bins=20, color='skyblue', edgecolor='black')
        axes[i].set_title(f"{param}")
        axes[i].set_xlabel(value_col[1].replace('_', ' ').title())
        axes[i].set_ylabel("Frequency")

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the suptitle
    return fig

def plot_3x1_histograms(data, param_list, title_prefix, value_col, fig_num):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle(f"{title_prefix} Histograms - Figure {fig_num}", fontsize=16)
    axes = axes.flatten()  # Flatten the 2x2 grid for easier iteration

    for i, param in enumerate(param_list[:4]):  # Take up to 4 parameters per figure
        param_data = data[data[value_col[0]] == param][value_col[1]]
        axes[i].hist(param_data, bins=20, color='skyblue', edgecolor='black')
        axes[i].set_title(f"{param}")
        axes[i].set_xlabel(value_col[1].replace('_', ' ').title())
        axes[i].set_ylabel("Frequency")

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the suptitle
    return fig


def analyze_and_plot_histograms(db_file_path='cascade_analysis.db', show_plots=True):
    # Connect to the database
    conn = sqlite3.connect(db_file_path)

    # Load data into pandas DataFrames
    validation_df = pd.read_sql_query("SELECT check_type, value FROM validation_checks", conn)
    test_results_df = pd.read_sql_query("""
        SELECT test_type, metric_1_name, metric_1_value, metric_2_name, metric_2_value 
        FROM test_results
    """, conn)

    # Close the connection
    conn.close()

    # Define parameters to plot
    validation_params = validation_df['check_type'].unique()  # e.g., Passivity, Reciprocity, Causality
    test_params = test_results_df['test_type'].unique()      # e.g., InsertionLoss, ReturnLoss, etc.

    # Plot validation checks (one figure)
    fig_num = 1
    plot_3x1_histograms(validation_df, validation_params, "Validation Checks", 
                        ["check_type", "value"], fig_num)
    '''
    # Plot test results (split across multiple figures if >4 test types)
    test_figs = []
    for i in range(0, len(test_params), 4):  # Step by 4 to fit 2x2 grid
        fig_num += 1
        fig = plot_2x2_histograms(test_results_df, test_params[i:i+4], "Test Results (Metric 1)", 
                                  ["test_type", "metric_1_value"], fig_num)
        test_figs.append(fig)

        fig_num += 1
        fig = plot_2x2_histograms(test_results_df, test_params[i:i+4], "Test Results (Metric 2)", 
                                  ["test_type", "metric_2_value"], fig_num)
        test_figs.append(fig)
    '''
    # Show all figures if requested
    if show_plots:
        plt.show()

def extract_and_sort_log_results(log_file: str = "LDV_log.log", output_dir: str = "S_params_out") -> bool:
    """
    Extract specific test results from LDV_log.log and create sorted output files.
    
    Args:
        log_file (str): Path to the input log file (default: LDV_log.log)
        output_dir (str): Directory for output files (default: S_params_out)
    
    Returns:
        bool: True if successful, False if an error occurs
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Keywords to filter log lines
        keywords = [
            "Causality check",
            "Passivity check",
            "Reciprocity check",
            "Test InsertionLoss",
            "Test ReturnLoss",
            "Test Max_ILD",
            "Test FOM_ILD",
            "Test D2C"
        ]
        
        # Initialize lists for each test type
        d2c_lines: List[Tuple[float, str]] = []
        fom_ild_lines: List[Tuple[float, str]] = []
        max_ild_lines: List[Tuple[float, str]] = []
        rl_lines: List[Tuple[float, str]] = []
        il_lines: List[Tuple[float, str]] = []
        general_results: List[str] = []
        
        # Regular expressions for extracting values
        d2c_pattern = r"D2C1: (-?\d+\.\d+) dB"
        fom_ild_pattern = r"FOM ILD: (-?\d+\.\d+) dB"
        max_ild_pattern = r"max deviation: (-?\d+\.\d+) dB"
        rl_pattern = r"RL1: (-?\d+\.\d+) dB"
        il_pattern = r"IL1: (-?\d+\.\d+) dB"
        
        # Read log file and process lines
        with open(log_file, 'r') as f:
            for line in f:
                # Check if line contains any keyword
                if any(keyword in line for keyword in keywords):
                    general_results.append(line.strip())
                    
                    # Process D2C test
                    if "Test D2C" in line:
                        match = re.search(d2c_pattern, line)
                        if match:
                            d2c1_value = float(match.group(1))
                            d2c_lines.append((d2c1_value, line.strip()))
                    
                    # Process FOM_ILD test
                    elif "Test FOM_ILD" in line:
                        match = re.search(fom_ild_pattern, line)
                        if match:
                            fom_ild_value = float(match.group(1))
                            fom_ild_lines.append((fom_ild_value, line.strip()))
                    
                    # Process Max_ILD test
                    elif "Test Max_ILD" in line:
                        match = re.search(max_ild_pattern, line)
                        if match:
                            max_deviation = float(match.group(1))
                            max_ild_lines.append((max_deviation, line.strip()))
                    
                    # Process ReturnLoss test
                    elif "Test ReturnLoss" in line:
                        match = re.search(rl_pattern, line)
                        if match:
                            rl1_value = float(match.group(1))
                            rl_lines.append((rl1_value, line.strip()))
                    
                    # Process InsertionLoss test
                    elif "Test InsertionLoss" in line:
                        match = re.search(il_pattern, line)
                        if match:
                            il1_value = float(match.group(1))
                            il_lines.append((il1_value, line.strip()))
        
        # Write general results to LDV_results.txt
        with open(os.path.join(output_dir, "LDV_results.txt"), 'w') as f:
            f.write("\n".join(general_results))
        
        # Write sorted D2C results
        d2c_lines.sort(key=lambda x: x[0], reverse=True)  # Sort by D2C1 (descending)
        with open(os.path.join(output_dir, "D2C_results.txt"), 'w') as f:
            f.write("\n".join(line for _, line in d2c_lines))
        
        # Write sorted FOM_ILD results
        fom_ild_lines.sort(key=lambda x: x[0], reverse=True)  # Sort by FOM ILD (descending)
        with open(os.path.join(output_dir, "ILD_FOM_results.txt"), 'w') as f:
            f.write("\n".join(line for _, line in fom_ild_lines))
        
        # Write sorted Max_ILD results
        max_ild_lines.sort(key=lambda x: x[0], reverse=True)  # Sort by max deviation (descending)
        with open(os.path.join(output_dir, "Max_FOM_results.txt"), 'w') as f:
            f.write("\n".join(line for _, line in max_ild_lines))
        
        # Write sorted ReturnLoss results
        rl_lines.sort(key=lambda x: x[0], reverse=True)  # Sort by RL1 (descending)
        with open(os.path.join(output_dir, "RL_results.txt"), 'w') as f:
            f.write("\n".join(line for _, line in rl_lines))
        
        # Write sorted InsertionLoss results
        il_lines.sort(key=lambda x: x[0], reverse=True)  # Sort by IL1 (descending)
        with open(os.path.join(output_dir, "IL_results.txt"), 'w') as f:
            f.write("\n".join(line for _, line in il_lines))
        
        print(f"Successfully created result files in {output_dir}")
        return True
    
    except FileNotFoundError as e:
        print(f"Error: Log file {log_file} not found: {str(e)}")
        return False
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False

def extract_cascade_files(log_file: str = "LDV_log.log", output_dir: str = "S_params_out") -> bool:
    """
    Extract cascade numbers and their file lists from LDV_log.log and write to cascade_files.txt.
    
    Args:
        log_file (str): Path to the input log file (default: LDV_log.log)
        output_dir (str): Directory for output file (default: S_params_out)
    
    Returns:
        bool: True if successful, False if an error occurs
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize list to store cascade data
        cascades: List[Tuple[int, str]] = []
        
        # Regular expression to extract cascade number
        cascade_pattern = r"Starting processing for cascade #(\d+)"
        files_pattern = r"Files in cascade: (\[.*?\])"
        
        # Read log file
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Process lines to find cascade numbers and file lists
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # Look for cascade start
            cascade_match = re.search(cascade_pattern, line)
            if cascade_match:
                cascade_num = int(cascade_match.group(1))
                # Look for the "Files in cascade" line within the next few lines
                for j in range(i + 1, min(i + 5, len(lines))):  # Check up to 5 lines ahead
                    files_match = re.search(files_pattern, lines[j])
                    if files_match:
                        files_list = files_match.group(1)
                        cascades.append((cascade_num, files_list))
                        break
                else:
                    print(f"Warning: No file list found for cascade #{cascade_num}")
            i += 1
        
        # Write to output file
        output_file = os.path.join(output_dir, "cascade_files.txt")
        with open(output_file, 'w') as f:
            f.write("cascade#, Files in cascade\n")
            for cascade_num, files_list in sorted(cascades, key=lambda x: x[0]):  # Sort by cascade number
                f.write(f"{cascade_num}, {files_list}\n")
        
        print(f"Successfully created {output_file} with {len(cascades)} cascades")
        return True
    
    except FileNotFoundError as e:
        print(f"Error: Log file {log_file} not found: {str(e)}")
        return False
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False
    


def create_worst_results_cascade(
    result_files: List[str] = [
        "RL_results.txt",
        "IL_results.txt",
        "D2C_results.txt",
        "ILD_FOM_results.txt",
        "Max_FOM_results.txt"
    ],
    cascade_file: str = "cascade_files.txt",
    output_dir: str = "S_params_out"
) -> bool:
    """
    Create worst_results_cascade.txt by extracting top 5 results from each result file,
    mapping their cascade indices to file lists from cascade_files.txt.
    
    Args:
        result_files (List[str]): List of result files to process
        cascade_file (str): Path to cascade_files.txt relative to output_dir
        output_dir (str): Directory containing result files and for output
    
    Returns:
        bool: True if successful, False if an error occurs
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Read cascade_files.txt to map cascade numbers to file lists
        cascade_map: Dict[int, str] = {}
        cascade_path = os.path.join(output_dir, cascade_file)
        if not os.path.exists(cascade_path):
            print(f"Error: {cascade_path} not found")
            return False
        
        with open(cascade_path, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            for line in lines:
                if line.strip():
                    try:
                        cascade_num, files_list = line.strip().split(",", 1)
                        cascade_map[int(cascade_num.strip())] = files_list.strip()
                    except ValueError:
                        print(f"Warning: Skipping malformed line in {cascade_file}: {line.strip()}")
        
        # Initialize list to store worst results
        worst_results: List[Tuple[int, str, str, str]] = []
        cascade_index_pattern = r"final_cascade_(\d+)\.s4p"
        
        # Process each result file
        for result_file in result_files:
            file_path = os.path.join(output_dir, result_file)
            test_type = result_file.replace("_results.txt", "")  # e.g., "RL" from "RL_results.txt"
            
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found, skipping")
                continue
            
            # Read top 5 lines
            with open(file_path, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                top_5 = lines[:5]  # Get top 5 (or fewer if file has less)
            
            # Extract cascade index from each line
            for line in top_5:
                match = re.search(cascade_index_pattern, line)
                if match:
                    cascade_num = int(match.group(1))
                    worst_results.append((cascade_num, test_type, line, cascade_map.get(cascade_num, "Files not found")))
                else:
                    print(f"Warning: No cascade index found in line: {line}")
        
        # Write to worst_results_cascade.txt
        output_file = os.path.join(output_dir, "worst_results_cascade.txt")
        with open(output_file, 'w') as f:
            f.write("Worst Performing Cascades\n")
            f.write("=" * 50 + "\n\n")
            for cascade_num, test_type, result_line, files_list in sorted(worst_results, key=lambda x: x[0]):
                f.write(f"Cascade #{cascade_num}\n")
                f.write(f"Test Type: {test_type}\n")
                f.write(f"Result: {result_line}\n")
                f.write(f"Files in Cascade: {files_list}\n")
                f.write("-" * 50 + "\n")
        
        print(f"Successfully created {output_file} with {len(worst_results)} worst results")
        return True
    
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False

def extract_worst_constellations(
    input_file: str = "worst_results_cascade.txt",
    output_dir: str = "S_params_out"
) -> bool:
    """
    Extract the worst-performing cascade constellations from worst_results_cascade.txt.
    
    Args:
        input_file (str): Path to worst_results_cascade.txt relative to output_dir
        output_dir (str): Directory for output file
    
    Returns:
        bool: True if successful, False if an error occurs
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        input_path = os.path.join(output_dir, input_file)
        
        if not os.path.exists(input_path):
            print(f"Error: {input_path} not found")
            return False
        
        # Data structures
        cascades: Dict[int, Dict] = {}  # cascade_num -> {test_type, result, files}
        metrics: Dict[str, List[Tuple[int, float, str]]] = {
            "IL": [], "RL": [], "D2C": [], "ILD_FOM": [], "Max_FOM": []
        }
        file_lists: List[str] = []  # Store file lists as strings for pattern analysis
        
        # Regular expressions
        cascade_pattern = r"Cascade #(\d+)"
        test_type_pattern = r"Test Type: (\w+)"
        result_pattern = r"Result: (.+)"
        files_pattern = r"Files in Cascade: (\[.*?\])"
        il_pattern = r"IL1: (-?\d+\.\d+) dB"
        rl_pattern = r"RL1: (-?\d+\.\d+) dB"
        d2c_pattern = r"D2C1: (-?\d+\.\d+) dB"
        fom_ild_pattern = r"FOM ILD: (\d+\.\d+) dB"
        max_dev_pattern = r"max deviation: (\d+\.\d+) dB"
        
        # Parse the file
        with open(input_path, 'r') as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                cascade_match = re.search(cascade_pattern, line)
                if cascade_match:
                    cascade_num = int(cascade_match.group(1))
                    if cascade_num not in cascades:
                        cascades[cascade_num] = {"tests": []}
                    
                    # Extract test type
                    i += 1
                    test_type_match = re.search(test_type_pattern, lines[i].strip())
                    if test_type_match:
                        test_type = test_type_match.group(1)
                        cascades[cascade_num]["tests"].append(test_type)
                    
                    # Extract result
                    i += 1
                    result_match = re.search(result_pattern, lines[i].strip())
                    if result_match:
                        result = result_match.group(1)
                        cascades[cascade_num]["result_" + test_type] = result
                        
                        # Extract metric value
                        if test_type == "IL":
                            match = re.search(il_pattern, result)
                            if match:
                                metrics["IL"].append((cascade_num, float(match.group(1)), result))
                        elif test_type == "RL":
                            match = re.search(rl_pattern, result)
                            if match:
                                metrics["RL"].append((cascade_num, float(match.group(1)), result))
                        elif test_type == "D2C":
                            match = re.search(d2c_pattern, result)
                            if match:
                                metrics["D2C"].append((cascade_num, float(match.group(1)), result))
                        elif test_type == "ILD_FOM":
                            match = re.search(fom_ild_pattern, result)
                            if match:
                                metrics["ILD_FOM"].append((cascade_num, float(match.group(1)), result))
                        elif test_type == "Max_FOM":
                            match = re.search(max_dev_pattern, result)
                            if match:
                                metrics["Max_FOM"].append((cascade_num, float(match.group(1)), result))
                    
                    # Extract files
                    i += 1
                    files_match = re.search(files_pattern, lines[i].strip())
                    if files_match:
                        files_list = files_match.group(1)
                        cascades[cascade_num]["files"] = files_list
                        file_lists.append(files_list)
                    
                    i += 2  # Skip separator
                else:
                    i += 1
        
        # Analyze frequent cascades
        cascade_counts = Counter(cascade_num for cascade_num in cascades)
        frequent_cascades = sorted(
            cascade_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]  # Top 5 most frequent
        
        # Analyze worst metrics
        worst_metrics = {}
        for test_type in metrics:
            metrics[test_type].sort(key=lambda x: x[1], reverse=True)  # Highest value first
            worst_metrics[test_type] = metrics[test_type][:1]  # Worst result per test type
        
        # Analyze common file patterns
        file_list_counts = Counter(file_lists)
        common_patterns = sorted(
            file_list_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]  # Top 5 most common file lists
        
        # Write to output file
        output_file = os.path.join(output_dir, "worst_constellations.txt")
        with open(output_file, 'w') as f:
            f.write("Worst Cascade Constellations Analysis\n")
            f.write("=" * 50 + "\n\n")
            
            # Section 1: Most Frequent Cascades
            f.write("Most Frequent Cascades (Appearing in Multiple Tests)\n")
            f.write("-" * 50 + "\n")
            for cascade_num, count in frequent_cascades:
                cascade = cascades[cascade_num]
                f.write(f"Cascade #{cascade_num} (Appears in {count} tests: {', '.join(cascade['tests'])})\n")
                f.write(f"Files: {cascade.get('files', 'Not found')}\n")
                for test_type in cascade["tests"]:
                    f.write(f"Result ({test_type}): {cascade.get('result_' + test_type, 'Not found')}\n")
                f.write("\n")
            
            # Section 2: Worst Metric Values
            f.write("Worst Metric Values per Test Type\n")
            f.write("-" * 50 + "\n")
            for test_type in worst_metrics:
                if worst_metrics[test_type]:
                    cascade_num, value, result = worst_metrics[test_type][0]
                    f.write(f"Test Type: {test_type}\n")
                    f.write(f"Cascade #{cascade_num}\n")
                    f.write(f"Value: {value} dB\n")
                    f.write(f"Result: {result}\n")
                    f.write(f"Files: {cascades[cascade_num].get('files', 'Not found')}\n")
                    f.write("\n")
            
            # Section 3: Common File Patterns
            f.write("Common File Patterns in Worst Cascades\n")
            f.write("-" * 50 + "\n")
            for file_list, count in common_patterns:
                f.write(f"File List (Appears {count} times):\n")
                f.write(f"{file_list}\n")
                # List cascades with this file list
                cascade_nums = [num for num, data in cascades.items() if data.get("files") == file_list]
                f.write(f"Cascades: {', '.join(map(str, cascade_nums))}\n")
                f.write("\n")
        
        print(f"Successfully created {output_file}")
        return True
    
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False

def create_cascade_metrics_table(
    input_file: str = "worst_results_cascade.txt",
    output_dir: str = "S_params_out"
) -> bool:
    """
    Create a CSV table with cascade metrics from worst_results_cascade.txt.
    
    Args:
        input_file (str): Path to worst_results_cascade.txt relative to output_dir
        output_dir (str): Directory for output file
    
    Returns:
        bool: True if successful, False if an error occurs
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        input_path = os.path.join(output_dir, input_file)
        
        if not os.path.exists(input_path):
            print(f"Error: {input_path} not found")
            return False
        
        # Data structures
        cascades: Dict[int, Dict] = defaultdict(lambda: {"files": [], "metrics": {}})
        max_file_count = 0
        
        # Regular expressions
        cascade_pattern = r"Cascade #(\d+)"
        test_type_pattern = r"Test Type: (\w+)"
        result_pattern = r"Result: (.+)"
        files_pattern = r"Files in Cascade: (\[.*?\])"
        il_pattern = r"IL1: (-?\d+\.\d+) dB, IL2: (-?\d+\.\d+) dB"
        rl_pattern = r"RL1: (-?\d+\.\d+) dB, RL2: (-?\d+\.\d+) dB"
        d2c_pattern = r"D2C1: (-?\d+\.\d+) dB, D2C2: (-?\d+\.\d+) dB"
        max_dev_pattern = r"max deviation: (\d+\.\d+) dB"
        fom_ild_pattern = r"FOM ILD: (\d+\.\d+) dB"
        
        # Parse the file
        with open(input_path, 'r') as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                cascade_match = re.search(cascade_pattern, line)
                if cascade_match:
                    cascade_num = int(cascade_match.group(1))
                    
                    # Extract test type
                    i += 1
                    test_type_match = re.search(test_type_pattern, lines[i].strip())
                    if test_type_match:
                        test_type = test_type_match.group(1)
                    
                    # Extract result and metrics
                    i += 1
                    result_match = re.search(result_pattern, lines[i].strip())
                    if result_match:
                        result = result_match.group(1)
                        if test_type == "IL":
                            match = re.search(il_pattern, result)
                            if match:
                                cascades[cascade_num]["metrics"]["IL1"] = float(match.group(1))
                                cascades[cascade_num]["metrics"]["IL2"] = float(match.group(2))
                        elif test_type == "RL":
                            match = re.search(rl_pattern, result)
                            if match:
                                cascades[cascade_num]["metrics"]["RL1"] = float(match.group(1))
                                cascades[cascade_num]["metrics"]["RL2"] = float(match.group(2))
                        elif test_type == "D2C":
                            match = re.search(d2c_pattern, result)
                            if match:
                                cascades[cascade_num]["metrics"]["D2C1"] = float(match.group(1))
                                cascades[cascade_num]["metrics"]["D2C2"] = float(match.group(2))
                        elif test_type == "ILD_FOM":
                            match = re.search(fom_ild_pattern, result)
                            if match:
                                cascades[cascade_num]["metrics"]["FOM ILD"] = float(match.group(1))
                        elif test_type == "Max_FOM":
                            match = re.search(max_dev_pattern, result)
                            if match:
                                cascades[cascade_num]["metrics"]["Max ILD"] = float(match.group(1))
                    
                    # Extract files
                    i += 1
                    files_match = re.search(files_pattern, lines[i].strip())
                    if files_match and files_match.group(1) != "Files not found":
                        # Parse file list string into individual files
                        files_str = files_match.group(1).strip("[]")
                        files = [f.strip().strip("'") for f in files_str.split(", ")]
                        cascades[cascade_num]["files"] = files
                        max_file_count = max(max_file_count, len(files))
                    
                    i += 2  # Skip separator
                else:
                    i += 1
        
        # Prepare CSV headers
        headers = ["Cascade name"]
        headers.extend([f"file{i+1} name" for i in range(max_file_count)])
        headers.extend(["IL1", "IL2", "RL1", "RL2", "D2C1", "D2C2", "Max ILD", "FOM ILD"])
        
        # Prepare CSV rows
        rows = []
        for cascade_num, data in sorted(cascades.items(), key=lambda x: x[0]):
            if not data["files"]:  # Skip cascades with no files
                continue
            row = [f"final_cascade_{cascade_num}"]
            # Add file names, padding with empty strings if fewer files
            row.extend(data["files"] + [""] * (max_file_count - len(data["files"])))
            # Add metrics, using "N/A" for missing values
            metrics = data["metrics"]
            for metric in ["IL1", "IL2", "RL1", "RL2", "D2C1", "D2C2", "Max ILD", "FOM ILD"]:
                row.append(str(metrics.get(metric, "N/A")))
            rows.append(row)
        
        # Write to CSV
        output_file = os.path.join(output_dir, "cascade_metrics_table.csv")
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        
        print(f"Successfully created {output_file} with {len(rows)} cascades")
        return True
    
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False
    
def create_cascade_metrics_table_from_log(
    log_file: str = "LDV_log.log",
    output_dir: str = "S_params_out"
) -> bool:
    """
    Create a CSV table with cascade metrics by parsing LDV_log.log.
    
    Args:
        log_file (str): Path to LDV_log.log
        output_dir (str): Directory for output file
    
    Returns:
        bool: True if successful, False if an error occurs
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        if not os.path.exists(log_file):
            print(f"Error: {log_file} not found")
            return False
        
        # Data structures
        cascades: Dict[int, Dict] = defaultdict(lambda: {"files": [], "metrics": {}})
        max_file_count = 0
        
        # Regular expressions
        cascade_pattern = r"Starting processing for cascade #(\d+)"
        files_pattern = r"Files in cascade: (\[.*?\])"
        il_pattern = r"Test InsertionLoss passed for final_cascade_\d+\.s4p with IL1: (-?\d+\.\d+) dB, IL2: (-?\d+\.\d+) dB"
        rl_pattern = r"Test ReturnLoss passed for final_cascade_\d+\.s4p with RL1: (-?\d+\.\d+) dB, RL2: (-?\d+\.\d+) dB"
        d2c_pattern = r"Test D2C passed for final_cascade_\d+\.s4p with D2C1: (-?\d+\.\d+) dB, D2C2: (-?\d+\.\d+) dB"
        max_dev_pattern = r"Test Max_ILD passed for final_cascade_\d+\.s4p with max deviation: (\d+\.\d+) dB"
        fom_ild_pattern = r"Test FOM_ILD passed for final_cascade_\d+\.s4p with.*FOM ILD: (\d+\.\d+) dB"
        
        # Parse the log file
        current_cascade = None
        with open(log_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Extract cascade number
                cascade_match = re.search(cascade_pattern, line)
                if cascade_match:
                    current_cascade = int(cascade_match.group(1))
                    if current_cascade not in cascades:
                        cascades[current_cascade] = {"files": [], "metrics": {}}
                    continue
                
                # Extract file list
                if current_cascade is not None:
                    files_match = re.search(files_pattern, line)
                    if files_match:
                        files_str = files_match.group(1).strip("[]")
                        files = [f.strip().strip("'") for f in files_str.split(", ")]
                        cascades[current_cascade]["files"] = files
                        max_file_count = max(max_file_count, len(files))
                
                # Extract metrics
                il_match = re.search(il_pattern, line)
                if il_match:
                    cascade_num = int(re.search(r"final_cascade_(\d+)\.s4p", line).group(1))
                    cascades[cascade_num]["metrics"]["IL1"] = float(il_match.group(1))
                    cascades[cascade_num]["metrics"]["IL2"] = float(il_match.group(2))
                
                rl_match = re.search(rl_pattern, line)
                if rl_match:
                    cascade_num = int(re.search(r"final_cascade_(\d+)\.s4p", line).group(1))
                    cascades[cascade_num]["metrics"]["RL1"] = float(rl_match.group(1))
                    cascades[cascade_num]["metrics"]["RL2"] = float(rl_match.group(2))
                
                d2c_match = re.search(d2c_pattern, line)
                if d2c_match:
                    cascade_num = int(re.search(r"final_cascade_(\d+)\.s4p", line).group(1))
                    cascades[cascade_num]["metrics"]["D2C1"] = float(d2c_match.group(1))
                    cascades[cascade_num]["metrics"]["D2C2"] = float(d2c_match.group(2))
                
                max_dev_match = re.search(max_dev_pattern, line)
                if max_dev_match:
                    cascade_num = int(re.search(r"final_cascade_(\d+)\.s4p", line).group(1))
                    cascades[cascade_num]["metrics"]["Max ILD"] = float(max_dev_match.group(1))
                
                fom_ild_match = re.search(fom_ild_pattern, line)
                if fom_ild_match:
                    cascade_num = int(re.search(r"final_cascade_(\d+)\.s4p", line).group(1))
                    cascades[cascade_num]["metrics"]["FOM ILD"] = float(fom_ild_match.group(1))
        
        # Prepare CSV headers
        headers = ["Cascade name"]
        headers.extend([f"file{i+1} name" for i in range(max_file_count)])
        headers.extend(["IL1", "IL2", "RL1", "RL2", "D2C1", "D2C2", "Max ILD", "FOM ILD"])
        
        # Prepare CSV rows
        rows = []
        for cascade_num, data in sorted(cascades.items(), key=lambda x: x[0]):
            if not data["files"]:  # Skip cascades with no files
                continue
            row = [f"final_cascade_{cascade_num}"]
            # Add file names, padding with empty strings if fewer files
            row.extend(data["files"] + [""] * (max_file_count - len(data["files"])))
            # Add metrics, using "N/A" for missing values
            metrics = data["metrics"]
            for metric in ["IL1", "IL2", "RL1", "RL2", "D2C1", "D2C2", "Max ILD", "FOM ILD"]:
                row.append(str(metrics.get(metric, "N/A")))
            rows.append(row)
        
        # Write to CSV
        output_file = os.path.join(output_dir, "cascade_metrics_table.csv")
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        
        print(f"Successfully created {output_file} with {len(rows)} cascades")
        return True
    
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False


# Optional: Run directly if this file is executed
if __name__ == "__main__":
    # Specify the path to your log file and optionally the database file
    
    db_file = "cascade_analysis.db"       # Optional custom database name

    # Run the database creation
    success = extract_cascade_files()
    print(f"Cascade file extraction completed with status: {'Success' if success else 'Failure'}")
    success = extract_and_sort_log_results()
    print(f"Extraction and sorting completed with status: {'Success' if success else 'Failure'}")
    success = create_worst_results_cascade()
    print(f"Worst results extraction completed with status: {'Success' if success else 'Failure'}")
    success = extract_worst_constellations()
    print(f"Worst constellations extraction completed with status: {'Success' if success else 'Failure'}")
    #success = create_cascade_metrics_table()
    #print(f"Cascade metrics table creation completed with status: {'Success' if success else 'Failure'}")
    success = create_cascade_metrics_table_from_log()
    print(f"Cascade metrics table creation completed with status: {'Success' if success else 'Failure'}")
    analyze_and_plot_histograms(db_file, show_plots=False)
    plot_test_type_histograms("InsertionLoss", db_file, show_plots=False)
    plot_test_type_histograms("ReturnLoss", db_file, show_plots=False)
    plot_test_type_histograms("D2C", db_file, show_plots=False)
    plot_max_ild_histograms(db_file, show_plots=True)
    print("Histogram analysis complete.")
    