import pandas as pd

# File path
file_path = r'C:\Users\kiran\OneDrive\Desktop\crypto prediction\Cryptocurrency-Price-Prediction\excel\dataset.csv'

# Load CSV into DataFrame
df = pd.read_csv(file_path)

print("-------------")
print(df)
print("-------------")

# Filter for Bitcoin
bitcoin_df = df[df['crypto_name'] == 'Bitcoin']

# Display the filtered DataFrame
print("Filtered DataFrame for Bitcoin:")
print(bitcoin_df)