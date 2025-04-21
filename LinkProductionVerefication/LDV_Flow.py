import json
from typing import Dict, List, Optional
import logging
from input_file_read import read_json_file
import os
from PRC_S_param_test import check_s_parameter_causality,check_s_parameter_passivity,check_s_parameter_reciprocity
import skrf as rf
import numpy as np
from s4p_self_cascading import cascade_s2p_or_s4p
from Plot_and_compare import accumalte_plot_s_parameters
import shutil
import time
import matplotlib.pyplot as plt
from network_tests import insertion_loss_test, return_loss_test, diffrential2common_loss_test, max_ILD_test, effective_return_loss
from log_praser import build_cascade_database
from LDV_analyze import analyze_and_plot_histograms, plot_test_type_histograms,plot_max_ild_histograms, extract_cascade_files, extract_and_sort_log_results, extract_cascade_files, extract_and_sort_log_results, extract_worst_constellations, create_cascade_metrics_table_from_log, create_worst_results_cascade

class SParameterValidator:
    def __init__(self, config_file: str, test_file: str,freq_rate: float = 26.5625e9,max_iterations: int = 10):
        self.config_file = config_file
        self.test_file = test_file
        self.max_iterations = max_iterations
        self.freq_rate = freq_rate
        self.config_data = None
        self.test_data = None
        self.s_param_files = []
        self.port_assignments = {}
        self.current_iteration = 0
        self.PASSIVITY_TOL = 1E-6
        self.RECIPROCITY_THRESH = 3  # 1 dB      
        self.CAUSALITY_THRESH = 0.7  # 1%
        
        # Set up logging
        logging.basicConfig(
            filename='LDV_log.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _log_and_return_false(self, error_msg: str) -> bool:
        """Helper method to log error and return False"""
        self.logger.error(error_msg)
        return False

    def run_validation_flow(self) -> bool:
        """Main method to execute the validation flow"""
        plot_enabled = 0
        try:
            # Start
            if not self._read_and_validate_config():
                print("FAIL: test struct validation failed")
                return False
            
            if not self._read_and_validate_test():
                print("FAIL: test TH validation failed")
                return False

            if not self._validate_s_param_files_exist():
                print("FAIL: S-parameter files validation failed")
                return False

            if not self._validate_s4p_files():
                print("FAIL: S4P file properties validation failed")
                return False
            # Generate all possible cascades and print them
            all_cascades = self.print_all_cascades()
            
            # Filter based on constraints
            valid_cascades = validator.filter_constrained_cascades(all_cascades)

            # Print results
            print(f"\nFound {len(valid_cascades)} valid cascade combinations after constraints:")
            
            if not self._wrap_process_cascade(valid_cascades):
                print("FAIL: wrap Cascade processing failed")
                return False
            
            # In run_validation_flow method, add this before the final return True:
            if not self._validate_final_cascades():
                print("FAIL: Final cascade validation failed - continue test")
                #return False
            
            if not self._test_method_process():
                print("FAIL: Test method process failed - continue test")
                #return False
            
            plt.show()
            return True

        except Exception as e:
            print(f"ERROR: Unexpected error in validation flow - {str(e)}")
            return False

    def _read_and_validate_config(self) -> bool:
        """Read and validate the configuration file (Step 1)"""
        
        try:
            #self.config_data = read_json_file(self.config_file)
            with open(self.config_file) as f:
                self.config_data = json.load(f)
            # Validate root structure
            if not isinstance(self.config_data, dict):
                return self._log_and_return_false("read_and_validate_config:Config file must be a JSON object (dict)")
            self.logger.info("read_and_validate_config:The file is a JSON object (dict)")
            if 'blocks' not in self.config_data:
                return self._log_and_return_false("read_and_validate_config:Missing required 'blocks' field in config")
            self.logger.info("read_and_validate_config:blocks exists in the config file")
            # Validate blocks structure
            if not isinstance(self.config_data['blocks'], list):
                return self._log_and_return_false("read_and_validate_config:'blocks' must be an array")
            self.logger.info("read_and_validate_config:blocks is an array")    
            for i, block in enumerate(self.config_data['blocks'], 1):
                if not isinstance(block, dict):
                    return self._log_and_return_false(f"read_and_validate_config:Block {i} must be an object, got {type(block)}")
                self.logger.info(f"read_and_validate_config:Block {i} is an object")
                
                if 'name' not in block:
                    return self._log_and_return_false(f"read_and_validate_config:Block {i} missing required 'name' field")
                self.logger.info(f"read_and_validate_config:Block {i} has a name")
                
                if 'files' not in block and 'constraint' not in block:
                    return self._log_and_return_false(f"read_and_validate_config:Block {i} must have either 'files' or 'constraint'")
                self.logger.info(f"read_and_validate_config:Block {i} has files or constraint")
                
                for j, file_entry in enumerate(block['files'], 1):
                    if not isinstance(file_entry, dict):
                        return self._log_and_return_false(
                            f"read_and_validate_config:File entry {j} in block {i} must be an object, got {type(file_entry)}")
                    self.logger.info(f"read_and_validate_config:File entry {j} in block {i} is an object")

                    # Mandatory file fields
                    if 'name' not in file_entry:
                        return self._log_and_return_false(
                            f"read_and_validate_config:File entry {j} in block {i} missing required 'name' field")
                    self.logger.info(f"read_and_validate_config:File entry {j} in block {i} has a name")

                    # port_assignment validation
                    if file_entry['port_assignment'] not in ['12>>34', '13>>24', '1>>2']:
                        return self._log_and_return_false(
                                    f"read_and_validate_config:File entry {j} in block {i} has invalid port_assignment " +
                                    f"'{file_entry['port_assignment']}'. Must be '12<<34', '13<<24', or 'n1>>2ot'")
                    self.logger.info(f"read_and_validate_config:File entry {j} in block {i} has valid port_assignment")
                    
                    if ('constraint' in file_entry and 
                        file_entry['constraint'] is not None and 
                        not isinstance(file_entry['constraint'], str)):
                        return self._log_and_return_false(
                            f"read_and_validate_config:File entry {j} in block {i} constraint must be string or null {file_entry}")
                    self.logger.info(f"read_and_validate_config:File entry {j} in block {i} has a valid constraint {file_entry}")
                    
                    
                    # Constraint validation
                    if ('constraint' in file_entry and 
                        file_entry['constraint'] is not None):
                        if 'constraint' in file_entry:
                            if 'constraint_type' not in file_entry:
                                return self._log_and_return_false(
                                    f"read_and_validate_config:File entry {j} in block {i} has constraint but missing 'constraint_type'")
                            self.logger.info(f"read_and_validate_config:File entry {j} in block {i} has constraint_type")

                            if file_entry['constraint_type'] not in ['must', 'must_not']:
                                return self._log_and_return_false(
                                    f"read_and_validate_config:File entry {j} in block {i} has invalid constraint_type " +
                                    f"'{file_entry['constraint_type']}'. Must be 'yes', 'no', or 'not'")
                            self.logger.info(f"read_and_validate_config:File entry {j} in block {i} has valid constraint_type")
            
            self.s_param_files = self._extract_s_param_files_from_config()
            self.port_assignments = self._extract_port_assignments()
            self.logger.info("read_and_validate_config:Config file validation passed successfully")
            print(f"Test_struct file validation passed successfully. Found {len(self.s_param_files)} S-parameter files.")
            for filename in self.s_param_files:
                self.logger.info(f"Existing S-parameter - {filename}")
            return True
            
        except json.JSONDecodeError as e:
            return self._log_and_return_false(f"Invalid Test_struct JSON format: {str(e)}")
        except Exception as e:
            return self._log_and_return_false(f"Unexpected error reading Test_struct: {str(e)}")

    def _read_and_validate_test(self) -> bool:
        
        """Load and validate the entire test configuration file"""
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Starting reading Test_TH.json")
        try:
            with open(self.test_file) as f:
                data = json.load(f)
            # Validate root structure
            
            if not isinstance(self.config_data, dict):
                return self._log_and_return_false("read_and_validate_test:test file must be a JSON object (dict)")
            self.logger.info("read_and_validate_test:The test file is a JSON object (dict)")
            # Check if 'tests' exists and is a list
            if 'tests' not in data or not isinstance(data['tests'], list):
                return self._log_and_return_false("read_and_validate_test:Missing required 'tests' field in config")
            self.logger.info("read_and_validate_test:tests exists in the test file")
            # Check each test in the tests list
            for test in data['tests']:
                # 1. Check test has 'name'
                if 'name' not in test or not isinstance(test['name'], str):
                    return self._log_and_return_false("read_and_validate_test:Missing required 'name' field in test")
                self.logger.info(f"read_and_validate_test:Test {test['name']} has a name")
                # 2. Check test has 'frequency'
                if 'frequency' not in test:
                    return self._log_and_return_false(f"read_and_validate_test:Missing required 'frequency' field in test {test['name']}")
                self.logger.info(f"read_and_validate_test:Test {test['name']} has a frequency")
                freq = test['frequency']
                
                # 3. Check frequency has either 'specific' or 'range' (but not both)
                if not (('specific' in freq) ^ ('range' in freq)):  # XOR operation
                    return self._log_and_return_false(f"read_and_validate_test:Missing either 'specific' or 'range' in 'frequency' field in test {test['name']}")
                self.logger.info(f"read_and_validate_test:Test {test['name']} has a valid frequency")
                # Validate the frequency value
                if 'specific' in freq:
                    if not isinstance(freq['specific'], (int, float)):
                        return self._log_and_return_false(f"read_and_validate_test:'frequency' field in test {test['name']} - is not int or float")
                    self.logger.info(f"read_and_validate_test:Test {test['name']} has a valid specific frequency")
                else:  # has 'range'
                    if not isinstance(freq['range'], dict):
                        return self._log_and_return_false(f"read_and_validate_test:'frequency' field in test {test['name']} - is not a object (dict)")
                    self.logger.info(f"read_and_validate_test:'frequency' field in test {test['name']} is a dict")
                    if not all(key in freq['range'] for key in ['min', 'max']):
                        return self._log_and_return_false(f"read_and_validate_test:'frequency' field in test {test['name']} - the range is missing min or max")
                    self.logger.info(f"read_and_validate_test:'frequency' field in test {test['name']} has min and max")
                    if not all(isinstance(freq['range'][key], (int, float)) for key in ['min', 'max']):
                        return self._log_and_return_false(f"read_and_validate_test:'frequency' field in test {test['name']} - the range min or max is not int or float")
                    self.logger.info(f"read_and_validate_test:'frequency' field in test {test['name']} - the range min and max are int or float")
                    if freq['range']['min'] > freq['range']['max']:
                        return self._log_and_return_false(f"read_and_validate_test:'frequency' field in test {test['name']} - min is greater than max")
                    self.logger.info(f"read_and_validate_test:'frequency' field in test {test['name']} - min is less than max")
                
                # Check test has 'th_limit'
                if 'th_limit' not in test or not isinstance(test['th_limit'], (int, float)):
                    return self._log_and_return_false(f"read_and_validate_test:In test {test['name']} - there is no th_limit")
                self.logger.info(f"read_and_validate_test:Test {test['name']} has a th_limit")
            # If all validations pass, save the data
            self.test_data = data
            self.logger.info("read_and_validate_test:Test_TH file validation passed successfully")
            print(f"Test_TH file validation passed successfully. Found {len(self.test_data['tests'])} tests.")
            return True
            
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading Test_TH file JSON format: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error loading Test_TH:: {e}")
            return False

    def _validate_s_param_files_exist(self) -> bool:
        """Validate all S-parameter files exist (Step 2)"""
        # Implementation would check each file exists in the filesystem
        try:
            missing_files = []
            inconsistent_files = []
            s_param_in_path = os.path.join('.', 'S_params_in')
            
            # First collect all unique file names from the config
            all_required_files = set()
            for block in self.config_data['blocks']:
                for file_entry in block['files']:
                    all_required_files.add(file_entry['name'])
                    # Also check files referenced in constraints if they exist
                    if 'constraint' in file_entry and file_entry['constraint'] is not None:
                        constraint_files = [f.strip() for f in file_entry['constraint'].split(',')]
                        all_required_files.update(constraint_files)
            
            # Check each file exists in S_param_in folder
            for filename in all_required_files:
                file_path = os.path.join(s_param_in_path, filename)
                if not os.path.exists(file_path):
                    missing_files.append(filename)
            
            if missing_files:
                error_msg = (
                    f"Missing {len(missing_files)} required S-parameter files in .\\S_param_in:\n" +
                    "\n".join(f"- {f}" for f in sorted(missing_files)) +
                    f"\nExpected path: {os.path.abspath(s_param_in_path)}"
                )
                return self._log_and_return_false(error_msg)
                
            self.logger.info(f"All {len(all_required_files)} required S-parameter files exist in .\\S_param_in")
            # =================
            # Additional tests for frequency consistency
            frequency_lengths = {}
            start_frequencies = {}
            end_frequencies = {}
            
            # Load each S-parameter file and collect frequency data
            for filename in all_required_files:
                file_path = os.path.join(s_param_in_path, filename)
                try:
                    ntwk = rf.Network(file_path)
                    freq = ntwk.f  # Frequency array in Hz
                    frequency_lengths[filename] = len(freq)
                    start_frequencies[filename] = freq[0]
                    end_frequencies[filename] = freq[-1]
                except Exception as e:
                    inconsistent_files.append((filename, f"Failed to read frequency data: {str(e)}"))
            
            # Test 1: Check if all S-parameter files have the same number of frequency points
            # It converts the values of frequency_lengths (the lengths) to a set to identify unique lengths.
            # If the set contains more than one value (len(set(frequency_lengths.values())) > 1), it means 
            # the files have different numbers of frequency points, and the test fails.
            if len(set(frequency_lengths.values())) > 1:
                error_msg = "S-parameter files have inconsistent frequency point counts:\n"
                for fname, length in frequency_lengths.items():
                    error_msg += f"  - {fname}: {length} points\n"
                    self.logger.error(error_msg)
                inconsistent_files.append(("Frequency Length", error_msg))
            self.logger.info(f"All S-parameter files have the same number of frequency points")
            # Test 2: Check if all S-parameter files start at the same frequency
            if len(set(start_frequencies.values())) > 1:
                error_msg = "S-parameter files have inconsistent start frequencies:\n"
                for fname, freq in start_frequencies.items():
                    error_msg += f"  - {fname}: {freq:.2e} Hz\n"
                    self.logger.error(error_msg)
                inconsistent_files.append(("Start Frequency", error_msg))
            self.logger.info(f"All S-parameter files have the same start frequency")
            # Test 3: Check if all S-parameter files end at the same frequency
            if len(set(end_frequencies.values())) > 1:
                error_msg = "S-parameter files have inconsistent end frequencies:\n"
                for fname, freq in end_frequencies.items():
                    error_msg += f"  - {fname}: {freq:.2e} Hz\n"
                    self.logger.error(error_msg)
                inconsistent_files.append(("End Frequency", error_msg))
            self.logger.info(f"All S-parameter files have the same end frequency")
            if inconsistent_files:
                error_msg = "S-parameter Frequency Validation Failures:\n"
                for test_name, msg in inconsistent_files:
                    error_msg += f"\n{test_name}:\n{msg}"
                return self._log_and_return_false(error_msg)
            
            self.logger.info(f"All {len(all_required_files)} S-parameter files have consistent frequency properties")
            # =================

            return True
        
        except Exception as e:
            return self._log_and_return_false(f"Error validating S-parameter files: {str(e)}")
        

    def _validate_s4p_files(self) -> bool:
        """Validate S4P files properties (Step 3)"""
        invalid_files = []
        s_param_in_path = os.path.join('.', 'S_params_in')
        for filepath in self.s_param_files:
            if not filepath.lower().endswith('.s4p'):
                continue
                
            try:
                filename = os.path.basename(filepath)
                file_path = os.path.join(s_param_in_path, filename)
                ntwk = rf.Network(file_path)
                filename = os.path.basename(filepath)
                valid = True
                validation_results = {}
                
                # 1. Passivity Check
                max_eig, _ = check_s_parameter_passivity(ntwk, filename)
                if np.any(max_eig > 1.0 + self.PASSIVITY_TOL):
                    validation_results['passivity'] = f"Failed (max eigenvalue: {np.max(max_eig):.6f})"
                    self.logger.error(f"Passivity check failed for {filename}= {np.max(max_eig):.2f}")
                    valid = False
                else:
                    self.logger.info(f"Passivity check passed for {filename}= {np.max(max_eig):.2f}")
                    validation_results['passivity'] = "Passed"
                
                # 2. Reciprocity Check
                _, max_dev = check_s_parameter_reciprocity(ntwk, filename,percentage_threshold=self.RECIPROCITY_THRESH)
                if np.any(max_dev > self.RECIPROCITY_THRESH):
                    validation_results['reciprocity'] = f"Failed (max deviation: {np.max(max_dev):.2f} dB)"
                    self.logger.error(f"reciprocity check failed for {filename}= {np.max(max_dev):.2f}")
                    valid = False
                else:
                    self.logger.info(f"reciprocity check passed for {filename}= {np.max(max_dev):.2f}")
                    validation_results['reciprocity'] = "Passed"
                
                # 3. Causality Check
                impulse_responses, nc_pct1, t = check_s_parameter_causality(network=ntwk, name=filename,percentage_threshold=self.CAUSALITY_THRESH,port_assignment=self.port_assignments[filepath])
                #print(max(nc_pct1.values()))
                if max(nc_pct1.values()) > self.CAUSALITY_THRESH:
                    validation_results['causality'] = f"Failed (max non-causal: {max(nc_pct1.values()):.2%})"
                    self.logger.error(f"causality check failed for {filename}= {max(nc_pct1.values()):.2f} limit is {self.CAUSALITY_THRESH:.2f}")
                    valid = False
                else:
                    self.logger.info(f"causality check passed for {filename}= {max(nc_pct1.values()):.2f}")
                    validation_results['causality'] = "Passed"
                
                if not valid:
                    invalid_files.append((filename, validation_results))
                    
            except Exception as e:
                invalid_files.append((os.path.basename(filepath), 
                                {"error": f"Processing failed: {str(e)}"}))
        
        if invalid_files:
            error_msg = "S4P Validation Failures:\n"
            for filename, results in invalid_files:
                error_msg += f"\nFile: {filename}\n"
                if 'error' in results:
                    error_msg += f"  - ERROR: {results['error']}\n"
                else:
                    for test, result in results.items():
                        error_msg += f"  - {test.upper()}: {result}\n"
            return self._log_and_return_false(error_msg)
        
        s4p_count = len([f for f in self.s_param_files if f.lower().endswith('.s4p')])
        self.logger.info(f"All {s4p_count} .s4p files passed passivity, reciprocity, and causality checks")
        return True

    def _process_iteration(self) -> bool:
        """Process a single iteration (Steps 4-6)"""
        # Generate S_param_out (implementation specific)
        s_param_out = self._generate_s_param_out()
        
        # Cascade all S-parameters
        cascaded_params = self._cascade_s_params(s_param_out)
        
        # Validate cascaded parameters
        if not self._validate_s4p_files(cascaded_params):
            return False
            
        return True

    def _final_validation_and_test(self) -> bool:
        """Final validation and test (Step 7)"""
        # Perform final cascade test
        final_result = self._test_cascade()
        
        # Save results
        with open('Test_results.txt', 'w') as f:
            f.write(str(final_result))
            
        return True
    # Placeholder methods for actual implementations
    def _extract_s_param_files_from_config(self) -> List[str]:
        """Extract S-parameter file paths from the new config structure"""
        s_param_files = []
        
        for block in self.config_data['blocks']:
            # Add main block files
            for file_entry in block.get('files', []):
                s_param_files.append(file_entry['name'])
                
            '''# Add constrained files if they exist
            for file_entry in block.get('files', []):
                if file_entry.get('constraint'):
                    constrained_files = [f.strip() for f in file_entry['constraint'].split(',')]
                    s_param_files.extend(constrained_files)
                    
        # Also add direct file constraints from blocks that aren't in 'files' array
        for block in self.config_data['blocks']:
            if block.get('constraint'):
                constrained_files = [f.strip() for f in block['constraint'].split(',')]
                s_param_files.extend(constrained_files)'''
                
        # Remove duplicates and return
        return list(set(s_param_files))

    def _extract_port_assignments(self) -> Dict[str, str]:
        #Extract port assignments for all files
        port_assignments = {}
        for block in self.config_data['blocks']:
            for file_entry in block['files']:
                port_assignments[file_entry['name']] = file_entry['port_assignment']
        return port_assignments

    def _generate_s_param_out(self) -> Dict:
        """Generate output S-parameters for current iteration"""
        return {}

    def _cascade_s_params(self, s_param_out: Dict) -> Dict:
        """Cascade all S-parameters including the new output"""
        return {}

    def _test_cascade(self) -> Dict:
        """Perform final test on cascaded S-parameters"""
        return {}
    
    def generate_all_cascades(self) -> List[List[Dict]]:
        """
        Generate all possible cascade combinations from the config structure,
        accounting for repeat counts for each block.
        Returns a list of cascades, where each cascade is a list of file entries.
        """
        cascades = [[]]  # Start with an empty cascade
        
        for block in self.config_data['blocks']:
            new_cascades = []
            repeat_count = block.get('repeat#', 1)  # Default to 1 if not specified
            
            # For each existing cascade, extend it with each file in current block
            for cascade in cascades:
                if 'files' in block and block['files']:
                    for file_entry in block['files']:
                        # Generate all possible repeat combinations (1 to repeat_count)
                        for repeat in range(1, repeat_count + 1):
                            new_cascade = cascade.copy()
                            # Add the file 'repeat' times
                            for _ in range(repeat):
                                new_cascade.append({
                                    'block_name': block['name'],
                                    'file_name': file_entry['name'],
                                    'port_assignment': file_entry['port_assignment'],
                                    'file_entry': file_entry  # Keep full entry for constraints
                                })
                            new_cascades.append(new_cascade)
                else:
                    # If block has no files, just pass through existing cascades
                    new_cascades.append(cascade.copy())
            
            cascades = new_cascades
        
        # Filter out empty cascades if any
        return [c for c in cascades if c]

    def print_all_cascades(self):
        """Print all possible cascade combinations in a readable format"""
        cascades = self.generate_all_cascades()
        
        print(f"\nFound {len(cascades)} possible cascade combinations:")
        for i, cascade in enumerate(cascades, 1):
            cascade_str = " -> ".join(
                f"{entry['block_name']}.{entry['file_name']}"
                for entry in cascade
            )
            self.logger.info(f"\nUnfilter cascade: {(cascade_str)}")
            #print(f"{i:3d}: {cascade_str}")
        
        return cascades

    def filter_constrained_cascades(self, cascades: List[List[Dict]]) -> List[List[Dict]]:
        """Filter cascades based on constraints in the config"""
        valid_cascades = []
        
        for cascade_idx, cascade in enumerate(cascades):
            valid = True
            self.logger.info(f"\nEvaluating cascade {cascade_idx + 1}: {[e['file_name'] for e in cascade]}")
            
            for i, entry in enumerate(cascade):
                file_entry = entry['file_entry']
                
                if 'constraint' in file_entry and file_entry['constraint']:
                    constraint_files = [f.strip() for f in file_entry['constraint'].split(',')]
                    constraint_type = file_entry.get('constraint_type', 'must')
                    current_file = entry['file_name']
                    
                    self.logger.info(f"\n  Checking constraints for {current_file} (position {i})")
                    self.logger.info(f"  Constraint type: {constraint_type}")
                    self.logger.info(f"  Constraint files: {constraint_files}")
                    self.logger.info(f"  Files after current: {[e['file_name'] for e in cascade[i+1:]]}")
                    
                    if constraint_type == 'must':
                        # Only constraint files may appear AFTER this one
                        for future_file in [e['file_name'] for e in cascade[i+1:]]:
                            if future_file not in constraint_files:
                                self.logger.error(f"    FAIL: Found non-constraint file '{future_file}' after '{current_file}'")
                                valid = False
                                break
                        else:
                            self.logger.info("    PASS: All subsequent files are in constraint list")
                    
                    elif constraint_type == 'must_not':
                        # Constraint files may NOT appear AFTER this one
                        for future_file in [e['file_name'] for e in cascade[i+1:]]:
                            if future_file in constraint_files:
                                self.logger.error(f"    FAIL: Found forbidden file '{future_file}' after '{current_file}'")
                                valid = False
                                break
                        else:
                            self.logger.info("    PASS: No forbidden files appear after current file")
                    
                    if not valid:
                        self.logger.info(f"  Cascade invalidated by constraints on {current_file}")
                        break
            
            if valid:
                self.logger.info("  Cascade VALID")
                valid_cascades.append(cascade)
            else:
                self.logger.info("  Cascade INVALID")
        
        self.logger.info(f"\nFiltered {len(cascades)} cascades down to {len(valid_cascades)} valid cascades")
        for cascade in enumerate(valid_cascades, 1):
            self.logger.info(f"\nValid cascade: {(cascade)}")
        return valid_cascades

    def _process_cascade(self, cascade: List[Dict]) -> bool:
        """Process a single cascade by sequentially cascading all S-parameter files with detailed logging"""
        try:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Starting processing for cascade #{self.current_iteration}")
            self.logger.info(f"Cascade length: {len(cascade)} files")
            self.logger.info(f"Files in cascade: {[entry['file_name'] for entry in cascade]}")
            
            # Initialize paths
            input_dir = os.path.join('.', 'S_params_in')
            output_dir = os.path.join('.', 'S_params_out')
            os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
            
            
            # Initialize with first file in the cascade
            if not cascade:
                self.logger.error("Empty cascade provided - nothing to process")
                return self._log_and_return_false("Empty cascade provided")
                
            previous_network = None
            temp_files = []  # To track intermediate files for cleanup
            processing_start_time = time.time()
            
            for i, entry in enumerate(cascade, 1):
                file_name = entry['file_name']
                current_file = os.path.join(input_dir, file_name)
                self.logger.info(f"\nProcessing file {i}/{len(cascade)}: {file_name}")
                self.logger.info(f"Full path: {os.path.abspath(current_file)}")
                
                if not os.path.exists(current_file):
                    self.logger.error(f"File not found at: {os.path.abspath(current_file)}")
                    return self._log_and_return_false(f"File not found: {current_file}")
                
                if previous_network is None:
                    # First file in cascade - just load it
                    load_start = time.time()
                    try:
                        previous_network = rf.Network(current_file)
                        load_time = time.time() - load_start
                        self.logger.info(f"Successfully loaded initial file (Ports: {previous_network.nports})")
                        self.logger.info(f"Load time: {load_time:.2f} seconds")
                        continue
                    except Exception as e:
                        self.logger.error(f"Failed to load initial file: {str(e)}")
                        return False
                    
                # Generate temp output name (in output directory)
                temp_output = os.path.join(output_dir, f"temp_cascade_{self.current_iteration}_{i}")
                self.logger.info(f"Preparing to cascade with previous result (Ports: {previous_network.nports})")
                self.logger.info(f"Temporary output will be saved as: {temp_output}")
                
                # Log network information before cascading
                self.logger.info(f"Previous network:\n{previous_network.name}")
                self.logger.info(f"Current network to cascade:\n{rf.Network(current_file).name}")
                
                # Cascade current file with previous result
                try:
                    load_start = time.time()
                    
                    # Get full paths for cascading - ensure proper extension
                    if i == 2:  # First cascading operation
                        # Use the original filename with extension from the config
                        previous_file_path = os.path.join(input_dir, cascade[i-2]['file_name'])
                    else:
                        # For subsequent operations, use the path from the previous network
                        previous_file_path = previous_network.name
                        # Ensure the path has the correct extension
                        if not any(previous_file_path.lower().endswith(ext) for ext in ('.s2p', '.s4p')):  # Changed
                        #if not previous_file_path.lower().endswith(('.s2p', '.s4p')):
                            previous_file_path += f'.s{previous_network.nports}p'
                        if not os.path.exists(previous_file_path):
                            alt_path = os.path.join(output_dir, os.path.basename(previous_file_path))
                            if os.path.exists(alt_path):
                                previous_file_path = alt_path
                            else:
                                return self._log_and_return_false(f"Previous network file not found: {previous_file_path}")  # Changed

                    current_file_path = current_file
                    
                    self.logger.info(f"Cascading with paths:")
                    self.logger.info(f"  Previous file: {previous_file_path}")
                    self.logger.info(f"  Current file: {current_file_path}")
                    self.logger.info(f"  Output file: {temp_output}")
                    
                    cascade_start = time.time()
                    ntwk1, ntwk2, cascaded_file = cascade_s2p_or_s4p(
                        previous_file_path,
                        current_file_path,
                        output_name=temp_output
                    )
                    cascade_time = time.time() - cascade_start
                    temp_output = temp_output + f".s{previous_network.nports}p"
                    # Verify the output file exists at the expected location
                    if not os.path.exists(temp_output):
                        self.logger.error(f"Output file not found at eithr {temp_output} ")
                        return False

                    temp_files.append(temp_output)  # Track the file we actually used
                    self.logger.info(f"Successfully cascaded files (Time: {cascade_time:.2f} sec)")
                    self.logger.info(f"Output saved to: {temp_output}")

                    # Load for next iteration
                    previous_network = rf.Network(temp_output)
                    self.logger.info(f"previous network name for next loop:\n{previous_network.name}")
                    
                except Exception as e:
                    self.logger.error(f"Cascading failed at step {i} with error: {str(e)}", exc_info=True)
                    return self._log_and_return_false(
                        f"Failed to cascade {previous_file_path} and {file_name}: {str(e)}"
                    )
            
            # Save final result to output directory
            final_output = os.path.join(output_dir, f"final_cascade_{self.current_iteration}.s{previous_network.nports}p")
            save_start = time.time()
            previous_network.write(final_output)
            save_time = time.time() - save_start
            self.logger.info(f"\nFinal cascade result saved to: {final_output}")
            self.logger.info(f"File save time: {save_time:.2f} seconds")
            self.logger.info(f"Total processing time: {time.time() - processing_start_time:.2f} seconds")
            
            # Clean up temp files
            if temp_files:
                self.logger.info("Cleaning up temporary files...")
                for temp_file in temp_files:
                    try:
                        os.remove(temp_file)
                        self.logger.debug(f"Removed temp file: {temp_file}")
                    except Exception as e:
                        self.logger.warning(f"Could not remove temp file {temp_file}: {str(e)}")
            
            self.logger.info(f"Successfully completed processing cascade #{self.current_iteration}")
            self.logger.info(f"{'='*50}\n")
            return True
            
        except Exception as e:
            self.logger.error(f"Unexpected error processing cascade: {str(e)}", exc_info=True)
            return self._log_and_return_false(f"Error processing cascade: {str(e)}")
    
    def _wrap_process_cascade(self, valid_cascades: List[List[Dict]]) -> bool:
        # Process each valid cascade
        plot_enabled = 1
        start_time = time.time()
        cascade_times = []  # To track processing time per cascade
        for cascade_idx, cascade in enumerate(valid_cascades, 1):
            cascade_start_time = time.time()
            self.current_iteration = cascade_idx
            self.logger.info(f"\nProcessing cascade {cascade_idx}/{len(valid_cascades)}")
            if cascade_times:
                avg_time = sum(cascade_times) / len(cascade_times)
                remaining = avg_time * (len(valid_cascades) - cascade_idx)
                #print(f"Estimated time remaining: {remaining:.1f}s (~{remaining/60:.1f} minutes)")
                print(f"\rEstimated time remaining: {remaining:.1f}s (~{remaining/60:.1f} minutes): based {cascade_idx} file out of {len(valid_cascades)}: Files in cascade: {[entry['file_name'] for entry in cascade]}", end="", flush=True)
            
            if not self._process_cascade(cascade):
                print(f"FAIL: Cascade {cascade_idx} validation failed")
                return False
            # Calculate and store cascade processing time
            cascade_time = time.time() - cascade_start_time
            cascade_times.append(cascade_time)
            # Print completion time
            # Load the final cascaded result for plotting
            final_output = os.path.join('S_params_out', f'final_cascade_{cascade_idx}.s4p')
            if os.path.exists(final_output):
                ntwk = rf.Network(final_output)
                # Plot with current figure/axes and update them
                if plot_enabled:
                    if cascade_idx == 1:
                        plot_fig, plot_axes = accumalte_plot_s_parameters(ntwk, fig=None, axes=None, 
                                                        label=f'Cascade {cascade_idx}')
                    else:
                        plot_fig, plot_axes = accumalte_plot_s_parameters(ntwk, fig=plot_fig, axes=plot_axes, 
                                                        label=f'Cascade {cascade_idx}')
        #print(f"Completed in {sum(cascade_times):.2f}s")
        print(f"\nCompleted in {cascade_time:.2f}s (Avg: {sum(cascade_times)/len(cascade_times):.2f}s)")
        print("SUCCESS: All cascades validated successfully")
        if plot_enabled:
                if plot_axes is not None:
                    if isinstance(plot_axes, (list, np.ndarray)):
                        for ax in plot_axes:
                            if hasattr(ax, 'get_legend'):
                                legend = ax.get_legend()
                                if legend is not None:
                                    legend.remove()
                    elif hasattr(plot_axes, 'get_legend'):
                        legend = plot_axes.get_legend()
                        if legend is not None:
                            legend.remove()
                    plt.tight_layout()
        return True
    
    def _validate_final_cascades(self) -> bool:
        """Validate all final cascade S-parameter files in S_params_out folder"""
        invalid_files = []
        s_param_out_path = os.path.join('.', 'S_params_out')
        
        # Find all final cascade files
        final_cascade_files = [
            f for f in os.listdir(s_param_out_path) 
            if f.startswith('final_cascade_') and f.lower().endswith(('.s2p', '.s4p'))
        ]
        
        if not final_cascade_files:
            return self._log_and_return_false("No final cascade files found in S_params_out folder")
        
        self.logger.info(f"Found {len(final_cascade_files)} final cascade files to validate")
        
        for filename in final_cascade_files:
            try:
                file_path = os.path.join(s_param_out_path, filename)
                ntwk = rf.Network(file_path)
                valid = True
                validation_results = {}
                
                # 1. Passivity Check
                max_eig, _ = check_s_parameter_passivity(ntwk, filename)
                if np.any(max_eig > 1.0 + self.PASSIVITY_TOL):
                    validation_results['passivity'] = f"Failed (max eigenvalue: {np.max(max_eig):.6f})"
                    self.logger.error(f"Passivity check failed for {filename} = {np.max(max_eig):.2f}")
                    valid = False
                else:
                    self.logger.info(f"Passivity check passed for {filename} = {np.max(max_eig):.2f}")
                    validation_results['passivity'] = "Passed"
                
                # 2. Reciprocity Check
                _, max_dev = check_s_parameter_reciprocity(ntwk, filename, percentage_threshold=self.RECIPROCITY_THRESH)
                if np.any(max_dev > self.RECIPROCITY_THRESH):
                    validation_results['reciprocity'] = f"Failed (max deviation: {np.max(max_dev):.2f} dB)"
                    self.logger.error(f"Reciprocity check failed for {filename} = {np.max(max_dev):.2f}")
                    valid = False
                else:
                    self.logger.info(f"Reciprocity check passed for {filename} = {np.max(max_dev):.2f}")
                    validation_results['reciprocity'] = "Passed"
                
                # 3. Causality Check
                # For final cascades, we might not have port assignments, so we'll use a default
                port_assignment = '12>>34' if ntwk.nports == 4 else '1>>2'
                impulse_responses, nc_pct1, t = check_s_parameter_causality(
                    network=ntwk, 
                    name=filename,
                    percentage_threshold=self.CAUSALITY_THRESH,
                    port_assignment=port_assignment
                )
                
                if max(nc_pct1.values()) > self.CAUSALITY_THRESH:
                    validation_results['causality'] = f"Failed (max non-causal: {max(nc_pct1.values()):.2%})"
                    self.logger.error(f"Causality check failed for {filename} = {max(nc_pct1.values()):.2f}")
                    valid = False
                else:
                    self.logger.info(f"Causality check passed for {filename} = {max(nc_pct1.values()):.2f}")
                    validation_results['causality'] = "Passed"
                
                if not valid:
                    invalid_files.append((filename, validation_results))
                    
            except Exception as e:
                invalid_files.append((filename, {"error": f"Processing failed: {str(e)}"}))
        
        if invalid_files:
            error_msg = "Final Cascade Validation Failures:\n"
            for filename, results in invalid_files:
                error_msg += f"\nFile: {filename}\n"
                if 'error' in results:
                    error_msg += f"  - ERROR: {results['error']}\n"
                else:
                    for test, result in results.items():
                        error_msg += f"  - {test.upper()}: {result}\n"
            return self._log_and_return_false(error_msg)
        
        self.logger.info(f"All {len(final_cascade_files)} final cascade files passed validation checks")
        print(f"Final cascade validation passed successfully. Validated {len(final_cascade_files)} files.")
        return True

    def _test_method_process(self) -> bool:
        """Process all test methods on the final cascaded S-parameter files"""
        try:
            s_param_out_path = os.path.join('.', 'S_params_out')
            
            # Find all final cascade files
            final_cascade_files = [
                f for f in os.listdir(s_param_out_path) 
                if f.startswith('final_cascade_') and f.lower().endswith(('.s2p', '.s4p'))
            ]
            
            if not final_cascade_files:
                return self._log_and_return_false("test_method_process::No final cascade files found for testing")
            
            self.logger.info(f"\nStarting test processing on {len(final_cascade_files)} cascade files")
            print(f"\nStarting test processing on {len(final_cascade_files)} cascade files")
            
            # Process each test in the test configuration
            for test in self.test_data['tests']:
                test_name = test['name']
                self.logger.info(f"Processing test: {test_name}")
                print(f"*****     Processing test: {test_name}")
                # Determine frequency range for the test
                if 'specific' in test['frequency']:
                    freq = test['frequency']['specific']
                    freq_range = (freq, freq)  # Single frequency as a range
                else:
                    freq_range = (test['frequency']['range']['min'], test['frequency']['range']['max'])
                
                # Process each cascade file with this test
                for filename in final_cascade_files:
                    file_path = os.path.join(s_param_out_path, filename)
                    try:
                        ntwk = rf.Network(file_path)
                        
                        # Call the appropriate test method based on test name
                        if test_name == "InsertionLoss":
                            IL1,IL2,result = insertion_loss_test(ntwk, freq, test['th_limit'])
                            if result == True:
                                self.logger.info(f"Test {test_name} passed for {filename} with IL1: {IL1:.2f} dB, IL2: {IL2:.2f} dB, THreshold: {test['th_limit']} dB")
                            else:
                                self.logger.error(f"Test {test_name} failed for {filename} with IL1: {IL1:.2f} dB, IL2: {IL2:.2f} dB, THreshold: {test['th_limit']} dB")
                        elif test_name == "ReturnLoss":
                            RL1,RL2,result = return_loss_test(ntwk, freq, test['th_limit'])
                            if result == True:
                                self.logger.info(f"Test {test_name} passed for {filename} with RL1: {RL1:.2f} dB, RL2: {RL2:.2f} dB, THreshold: {test['th_limit']} dB")
                            else:
                                self.logger.error(f"Test {test_name} failed for {filename} with RL1: {RL1:.2f} dB, RL2: {RL2:.2f} dB, THreshold: {test['th_limit']} dB")
                        elif test_name == "Max_ILD":
                            max_deviation, min_deviation, avg_deviation, pass_fail_ILD,pass_fail_FOM, frequencies, deviations, fit_params,fom_ild_value = max_ILD_test(ntwk, self.freq_rate, freq_range[0],freq_range[1], TH_ILD = test['th_limit'],TH_FOM = 10,debug_mode=0)
                            if pass_fail_ILD == True:
                                self.logger.info(f"Test {test_name} passed for {filename} with max deviation: {max_deviation:.2f} dB,avg deviation: {avg_deviation:.2f} dB,FOM ILD: {fom_ild_value:.2f} dB, THreshold: {test['th_limit']} dB")
                            else:
                                self.logger.error(f"Test {test_name} failed for {filename} with max deviation: {max_deviation:.2f} dB,avg deviation: {avg_deviation:.2f} dB,FOM ILD: {fom_ild_value:.2f} dB, THreshold: {test['th_limit']} dB")
                        elif test_name == "FOM_ILD":
                            max_deviation, min_deviation, avg_deviation, pass_fail_ILD,pass_fail_FOM, frequencies, deviations, fit_params,fom_ild_valueresult =  max_ILD_test(ntwk, self.freq_rate, freq_range[0],freq_range[1], TH_ILD = 10, TH_FOM = test['th_limit'],debug_mode=0)
                            if pass_fail_ILD == True:
                                self.logger.info(f"Test {test_name} passed for {filename} with max deviation: {max_deviation:.2f} dB,avg deviation: {avg_deviation:.2f} dB,FOM ILD: {fom_ild_value:.2f} dB, THreshold: {test['th_limit']} dB")
                            else:
                                self.logger.error(f"Test {test_name} failed for {filename} with max deviation: {max_deviation:.2f} dB,avg deviation: {avg_deviation:.2f} dB,FOM ILD: {fom_ild_value:.2f} dB, THreshold: {test['th_limit']} dB")
                        elif test_name == "D2C":
                            d2c1,d2c2,result = diffrential2common_loss_test(ntwk, freq, test['th_limit'])
                            if result == True:
                                self.logger.info(f"Test {test_name} passed for {filename} with D2C1: {d2c1:.2f} dB, D2C2: {d2c2:.2f} dB, THreshold: {test['th_limit']} dB")
                            else:
                                self.logger.error(f"Test {test_name} failed for {filename} with D2C1: {d2c1:.2f} dB, D2C2: {d2c2:.2f} dB, THreshold: {test['th_limit']} dB")
                        elif test_name == "ERL":
                            result = effective_return_loss(ntwk, freq_range, test['th_limit'])
                        else:
                            self.logger.warning(f"Unknown test type: {test_name}")
                            continue
                        
                        # Log the test result
                        #self.log_test_result(filename, test_name, result, test['th_limit'])
                        
                    except Exception as e:
                        self.logger.error(f"Failed to process test {test_name} on file {filename}: {str(e)}")
                        return False
            
            self.logger.info("All tests processed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in test_method_process: {str(e)}")
            return False
        

if __name__ == "__main__":
    validator = SParameterValidator('Test_struct.json','Test_TH.json',freq_rate = 26.5625e9, max_iterations=5)
    success = True
    success = validator.run_validation_flow()
    print(f"Validation completed with status: {'Success' if success else 'Failure'}")

    # Specify the path to your log file and optionally the database file
    log_file = "LDV_log.log"  # Replace with actual path
    db_file = "cascade_analysis.db"       # Optional custom database name

    # Run the database creation
    build_cascade_database(log_file, db_file)
    success = extract_cascade_files()
    print(f"Cascade file extraction completed with status: {'Success' if success else 'Failure'}")
    success = extract_and_sort_log_results()
    print(f"Extraction and sorting completed with status: {'Success' if success else 'Failure'}")
    success = create_worst_results_cascade()
    print(f"Worst results extraction completed with status: {'Success' if success else 'Failure'}")
    success = extract_worst_constellations()
    print(f"Worst constellations extraction completed with status: {'Success' if success else 'Failure'}")
    success = create_cascade_metrics_table_from_log()
    print(f"Cascade metrics table creation completed with status: {'Success' if success else 'Failure'}")
    analyze_and_plot_histograms(db_file, show_plots=False)
    plot_test_type_histograms("InsertionLoss", db_file, show_plots=False)
    plot_test_type_histograms("ReturnLoss", db_file, show_plots=False)
    plot_test_type_histograms("D2C", db_file, show_plots=False)
    plot_max_ild_histograms(db_file, show_plots=True)
    print("Histogram analysis complete.")