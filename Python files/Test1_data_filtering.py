import pandas as pd

# File path
file_path = r'C:\Users\rooyv\Documents\Loyalist\STEP1\Crypto_data.csv'

# Load CSV into DataFrame
df = pd.read_csv(file_path)


# Filter for Bitcoin
bitcoin_df = df[df['crypto_name'] == 'Bitcoin']

# Display the filtered DataFrame
print("Filtered DataFrame for Bitcoin:")
print(bitcoin_df)