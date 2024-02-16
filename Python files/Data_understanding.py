import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# Read dataset and display basic information
df = pd.read_csv('dataset.csv')
print(df.head())
print(df.shape)
print(df.describe())

# Plot historical close price of Bitcoin
plt.figure(figsize=(10, 5))
plt.plot(df['close'])
plt.title('Bitcoin Close Price', fontsize=15)
plt.ylabel('Price in Dollars')
plt.show()

# Display total null values from all columns
print(df.isnull().sum())

# Define features for future use
features = ['open', 'high', 'low', 'close']
# Use subplots to visualize feature distributions
plt.subplots(figsize=(20, 10))
for i, col in enumerate(features):
    plt.subplot(2, 2, i + 1)
    sb.distplot(df[col])
plt.show()

# Extract Year, Month, and Day from the Date column
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
print(df.head())

# Group DataFrame by year and calculate mean for each group
data_grouped = df.groupby('year').mean()

# Plot average prices for each year's open, high, low, and close
plt.subplots(figsize=(20, 10))
for i, col in enumerate(['open', 'high', 'low', 'close']):
    plt.subplot(2, 2, i + 1)
    data_grouped[col].plot.bar()
plt.show()

# Boxplot for open price with outliers
plt.title('Boxplot of Crypto Open Prices (Includes Outliers)')
plt.xlabel('Open Price')
sb.boxplot(data=df['open'], showfliers=True, orient='h')
plt.show()
