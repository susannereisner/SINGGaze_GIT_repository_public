# -*- coding: utf-8 -*-
"""
SING Gaze Project
Susanne Reisner

This script reads .xlsx look files from a directory, deletes rows with
looks <1s long and saves them to a new directory.

The results of this study have been published as:
"The reciprocal relationship between maternal infant-directed singing and infant gaze"
in Musicae Scientiae, https://doi.org/10.1177/10298649251385676
"""

import os
import pandas as pd
# glob

# Define paths
source_dir = "" # insert your source directory with the relevant .xlsx files
target_dir = "" # where you want to save the cleaned .xlsx files

# Ensure target directory exists
os.makedirs(target_dir, exist_ok=True)

# Process each matching XLSX file
for file in os.listdir(source_dir):
    if file.endswith("_mum.xlsx"):
        file_path = os.path.join(source_dir, file)
        df = pd.read_excel(file_path)
        
        # Filter out rows where 'Duration' is < 1
        df_filtered = df[df['Duration'] >= 1]
        
        # Save to target directory
        output_path = os.path.join(target_dir, file)
        df_filtered.to_excel(output_path, index=False)
        print(f"Processed and saved: {output_path}")
