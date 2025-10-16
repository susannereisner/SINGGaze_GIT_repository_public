# -*- coding: utf-8 -*-
"""
SING Gaze Project, Wieki, University of Vienna
Script author: Susanne Reisner
February 2025

This script creates a file summarising .csv output of % of where infants looked
during each trial.

The results of this study have been published as:
"The reciprocal relationship between maternal infant-directed singing and infant gaze"
in Musicae Scientiae, https://doi.org/10.1177/10298649251385676
"""

import os
import pandas as pd

def process_csv_files(input_dirs, output_file):
    all_data = []
    
    for directory in input_dirs:
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            continue
        
        files_processed = 0
        for file in os.listdir(directory):
            if file.endswith(".csv"):
                file_path = os.path.join(directory, file)
                print(f"Processing file: {file_path}")
                
                try:
                    df = pd.read_csv(file_path, delimiter=";")
                    
                    expected_columns = {"ID", "Code", "condition","song_duration" ,"Frequency", "Duration", "Percentage over time", "Standard deviation of duration"}
                    actual_columns = set(df.columns)
                    
                    if not expected_columns.issubset(actual_columns):
                        print(f"Skipping {file} due to missing columns. Found columns: {actual_columns}")
                        continue
                    
                    # Extract ID
                    df["ID"] = df["ID"].astype(str)
                    
                    # # Sum total duration as 'song_duration'
                    # song_duration = df["Duration"].sum()
                    
                    # Pivot the table to reshape it
                    df_pivot = df.pivot(index=["ID", "condition", "song_duration"], columns="Code", values=["Frequency", "Duration", "Percentage over time", "Standard deviation of duration"])
                    # df_pivot.columns = [f"{col[1].lower()}_{col[0].split()[0].lower()}" for col in df_pivot.columns]
                    df_pivot.columns = [f"{str(col[1]).lower()}_{str(col[0]).split()[0].lower()}" for col in df_pivot.columns]
                    df_pivot.reset_index(inplace=True)
                    
                    # # Add total duration
                    # df_pivot.insert(2, "duration", song_duration)
                    
                    # Fill missing values with 0
                    df_pivot.fillna(0, inplace=True)
                    
                    all_data.append(df_pivot)
                    files_processed += 1
                except Exception as e:
                    print(f"Error processing {file}: {e}")
    
    if not all_data:
        print("No valid CSV files were processed.")
        return
    
    # Combine all data
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Save to CSV
    final_df.to_csv(output_file, index=False)
    print(f"Merged file saved to: {output_file}")

# Directories containing CSV files
input_dirs = [
    r"", # input files, named "ID.csv" (e.g., "1.csv"). files contain ID;look_location;condition;song_duration;look_Frequency;look_Duration;Percentage over time;Standard deviation of duration
    r"" # in case you have multiple folders with input files
]

# Output file path
output_file = r"" # where to save the output csv

# Run the processing function
process_csv_files(input_dirs, output_file)



