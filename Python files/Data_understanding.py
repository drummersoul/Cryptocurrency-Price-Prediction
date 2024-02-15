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

# Group the DataFrame 'df' by the 'year' column and calculate the mean of each numeric column for each group
#numeric_only is to calculate mean only for numbers
data_grouped = df.groupby(['year']).mean(numeric_only=True)
print(data_grouped)

# Create a new figure and subplots with a specific size (20x10 inches)
plt.subplots(figsize=(20, 10))

# Iterate over each column ('open', 'high', 'low', 'close') and its corresponding index
for i, col in enumerate(['open', 'high', 'low', 'close']):
    # Create subplots in a 2x2 grid, with each subplot representing one of the numeric columns
    plt.subplot(2, 2, i + 1)

    # Plot a bar chart for the current column using the grouped data
    data_grouped[col].plot.bar()

# Display the plot
plt.show()
# This are the box plots for open..similar can be done for close,low,high.
plt.title('This is a boxplot of Crypto Open Prices includes outliers')# This is the Title for Boxplot
plt.xlabel('open price') #label for open boxplot
sb.boxplot(data=df['open'], showfliers=True ,orient='h') #df reads column open ,showflies shows outliers and orientation will be horizontal(h) or vertical(v)
#Displays the plot
plt.show()