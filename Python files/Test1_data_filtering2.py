import pandas as pd

# File path
file_path = r'C:\Users\rooyv\Documents\Loyalist\STEP1\Crypto_data.csv'

# Load CSV into a DataFrame
df = pd.read_csv(file_path)

# List of cryptocurrencies to filter
crypto_list = ['Bitcoin', 'Litecoin', 'XRP']

# Create an empty list to store DataFrames
crypto_dfs = [] 

# Loop through each cryptocurrency
for crypto in crypto_list:
    # Filter for the current cryptocurrency
    crypto_df = df[df['crypto_name'] == crypto]

    # Calculate maximum and minimum close values
    max_close_value = crypto_df['close'].max()
    min_close_value = crypto_df['close'].min()

    # Create a new DataFrame with the results
    result_df = pd.DataFrame(
        {'Crypto': [crypto], 'Max Close Value': [max_close_value], 'Min Close Value': [min_close_value]})

    # Append the DataFrame to the list
    crypto_dfs.append(result_df)

# Concatenate the list of DataFrames into a single DataFrame
result_df = pd.concat(crypto_dfs, ignore_index=True)

# Display the result DataFrame
print("Maximum and Minimum Close Values for Bitcoin, Litecoin, and XRP:")
print(result_df)