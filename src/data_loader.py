import os
import pandas as pd

def load_movie_data(data_folder):
    """Load all CSV files in the specified folder into a single DataFrame."""
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]  # list all CSV files
    all_movies_df = pd.concat(
        (pd.read_csv(f"{data_folder}{file}") for file in csv_files),
        ignore_index=True
    )
    return all_movies_df
