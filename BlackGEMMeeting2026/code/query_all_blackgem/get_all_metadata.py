#!/usr/bin/env python3
import os
import subprocess
import pandas as pd
from astropy.table import Table

# Paths
FITS_PATH = os.path.expanduser("~/Documents/supercatalogue/super_catalogue_v9.fits")
QUERY_SCRIPT = "/home/fabian/Projects/query_fullsource/query_fullsource.py"
TEMP_IDS_FILE = "source_ids_temp.csv"
OUTPUT_FILE = "output_full.csv"

# Load and filter FITS table (exclude BHBs only)
print("Loading FITS table...")
fits_data = Table.read(FITS_PATH)

print("Filtering out BHBs...")
mask = fits_data['cat'] != 'BHB'
source_ids = fits_data[mask]['source_id'].data.tolist()
print(f"Total source_ids after filtering: {len(source_ids)}")

# Write source IDs to a temporary CSV for query_fullsource
print(f"Writing source IDs to {TEMP_IDS_FILE}...")
pd.DataFrame({'SOURCE_ID': source_ids}).to_csv(TEMP_IDS_FILE, index=False)

# Run query_fullsource with the file as input
print("Running query_fullsource...")
cmd = [
    "python", QUERY_SCRIPT,
    OUTPUT_FILE,
    "--source_ids", TEMP_IDS_FILE,
    "--output_type", "detections"
]
subprocess.run(cmd, check=True)

print(f"\nDone. Results written to {OUTPUT_FILE}")