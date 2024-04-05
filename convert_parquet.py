import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

# Function to read and write a single Parquet file with the desired compression                                                                                                                                                                                                           
def convert_parquet_file(parquet_file, compression='snappy'):
    df = pd.read_parquet(parquet_file)
    df.to_parquet(parquet_file, compression=compression)
    print(f"Done: {parquet_file}")

# Set the directory where your Parquet files are located                                                                                                                                                                                                                                  
parquet_directory = Path('/home/ck/Downloads/refinedweb')

# Get a list of all Parquet files                                                                                                                                                                                                                                                         
parquet_files = list(parquet_directory.glob('*.parquet'))

# Set the number of worker processes to the number of available CPU cores                                                                                                                                                                                                                 
num_workers = 16

# Execute the conversion in parallel using ProcessPoolExecutor                                                                                                                                                                                                                            
with ProcessPoolExecutor(max_workers=num_workers) as executor:
    # Map the convert_parquet_file function to each Parquet file                                                                                                                                                                                                                          
    futures = [executor.submit(convert_parquet_file, parquet_file) for parquet_file in parquet_files]
    # Wait for all futures to complete                                                                                                                                                                                                                                                    
    for future in as_completed(futures):
        try:
            future.result()  # You can handle exceptions here if needed                                                                                                                                                                                                                   
        except Exception as e:
            print(f"An error occurred: {e}")

print("Conversion completed.")
