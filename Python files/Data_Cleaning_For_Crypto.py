import os
import pandas as pd 

# Get the current working directory
current_directory = os.getcwd()

# Specify the relative path to the CSV file from the current directory
csv_file_location = os.path.join(current_directory, "Excel DB", "Crypto_data_info.csv")

# Check if the file exists
if os.path.exists(csv_file_location):
    df = pd.read_csv(csv_file_location)
    print("File successfully loaded.")
else:
    print(f"File not found at location: {csv_file_location}")

# Printing dataframe
print(df.head(10))
print(df)

#Average of whole close column
print(df["close"].mean())

#Check close if null
print(df["close"].isnull().sum())

#Check if any columns are null in dataset
print(df.isnull().sum())

#Get all unique crypto from crypto column and 
categoriesDf = df['crypto_name'].unique().tolist()
print(len(categoriesDf))
    