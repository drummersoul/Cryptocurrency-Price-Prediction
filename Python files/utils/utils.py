import os
import pandas as pd

def get_csv_file(file_name):
    # Get the current working directory
    current_directory = os.getcwd()

    # Specify the relative path to the CSV file from the current directory
    csv_file_location = os.path.join(current_directory, "Excel DB", file_name)

    return csv_file_location

def get_data(file_name):
    """return data as data frame"""
    path = get_csv_file(file_name)

    # Check if the file exists
    if os.path.exists(path):
        df = pd.read_csv(path)
        print("File successfully loaded.")
        dataframe = pd.read_csv(path)
        return dataframe
    else:
        print(f"File not found at location: {path}")
