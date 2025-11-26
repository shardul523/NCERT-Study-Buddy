import json
import glob
import os

def combine_raft_datasets(output_file="data/raft_dataset.jsonl", input_pattern="data/raft_dataset_*.jsonl"):
    """
    Combines multiple RAFT dataset JSONL files into a single file.
    """
    input_files = glob.glob(input_pattern)
    # Exclude the output file if it matches the pattern (though it shouldn't with the current naming)
    input_files = [f for f in input_files if os.path.abspath(f) != os.path.abspath(output_file)]
    
    print(f"Found {len(input_files)} files to combine: {input_files}")
    
    total_records = 0
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for input_file in input_files:
            print(f"Processing {input_file}...")
            try:
                with open(input_file, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        if line.strip(): # Skip empty lines
                            outfile.write(line)
                            total_records += 1
            except Exception as e:
                print(f"Error reading {input_file}: {e}")

    print(f"Successfully combined {total_records} records into {output_file}")

if __name__ == "__main__":
    # Ensure we are running from the project root or adjust paths
    if not os.path.exists("data"):
        print("Error: 'data' directory not found. Please run this script from the project root.")
    else:
        combine_raft_datasets()
