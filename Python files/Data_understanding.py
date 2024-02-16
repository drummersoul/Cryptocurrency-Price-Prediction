import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from utils.utils import get_data

# Read dataset and display basic information
df = get_data('Crypto_data_info.csv')
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

#This creates a figure and axis grid with a specific size of 20 inches in width and 10 inches in height.
plt.subplots(figsize=(20, 10))
# Iterate over each feature in the 'features' list
for i, col in enumerate(features):
    # Create subplots within the grid, with 2 rows, 2 columns, and index i + 1
    plt.subplot(2, 2, i + 1)
    # Plot a distribution plot (histogram and kernel density estimate) for the current feature
    #with this we can visualize the distribution of each feature data
    sb.distplot(df[col])
#This displays the plot grid created by the previous subplots.
plt.show()

#convting object [date] column format to dateTime format
df['date'] = pd.to_datetime(df['date']) 

# Extract the year from the 'date' column using the dt accessor in pandas
df['year'] = df['date'].dt.year

# Extract the month from the 'date' column using the dt accessor in pandas
df['month'] = df['date'].dt.month

# Extract the day from the 'date' column using the dt accessor in pandas
df['day'] = df['date'].dt.day

# Print the first few rows of the DataFrame to see the changes
print(df.head())

# Group the DataFrame 'df' by the 'year' column and calculate the mean of each numeric column for each group
#numeric_only is to calculate mean only for numbers
data_grouped = df.groupby(by=['year']).mean(numeric_only=True)
print(data_grouped)

# Create a new figure and subplots with a specific size (20x10 inches)
plt.subplots(figsize=(20, 10))

# Iterate over each column ('open', 'high', 'low', 'close') and its corresponding index
for i, col in enumerate(['open', 'high', 'low', 'close']):
    # Create subplots in a 2x2 grid, with each subplot representing one of the numeric columns
    plt.subplot(2, 2, i + 1)

    # Plot a bar chart for the current column using the grouped data
    data_grouped[col].plot.bar()

#Displays the plot
plt.show()

# This are the box plots for open..similar can be done for close,low,high.
plt.title('This is a boxplot of Crypto Open Prices includes outliers')# This is the Title for Boxplot
plt.xlabel('open price') #label for open boxplot
sb.boxplot(data=df['open'], showfliers=True ,orient='h') #df reads column open ,showflies shows outliers and orientation will be horizontal(h) or vertical(v)
#Displays the plot
plt.show()

#Correlation for Bitcoin crypto 
plt.figure(num="Correlation HeatMap For Bitcoin")
#Correlation for bitcoin crypto plt.figure(num="Correlation HeatMap For BitCoin")
corr = df.loc[df['crypto_name'] == 'Bitcoin'].iloc[:,1:].corr(method='spearman', numeric_only=True).round(2)
sb.heatmap(corr, annot=True)
plt.title("Correlation HeatMap for Bitcoin")

#Correlation for Litecoin crypto 
plt.figure(num="Correlation HeatMap For Litecoin")
corr = df.loc[df['crypto_name'] == 'Litecoin'].iloc[:,1:].corr(method='spearman', numeric_only=True).round(2)
sb.heatmap(corr, annot=True)
plt.title("Correlation HeatMap for Litecoin")

visualize_cols = ['open', 'high', 'low', 'volume', 'marketCap']
#ploting graph to check correlation
plt.figure(num="Scatter Plot")
for index, val in enumerate(visualize_cols):
    plt.subplot(3,2,index+1)
    plt.scatter(df.loc[df['crypto_name'] == 'Bitcoin'][val], df.loc[df['crypto_name'] == 'Bitcoin']['close'])

    # bestfit line logic
    # m, c = np.polyfit(df.loc[df['crypto_name'] == 'Bitcoin'][val], df.loc[df['crypto_name'] == 'Bitcoin']['marketCap'],deg= 1)
    # plt.plot(df.loc[df['crypto_name'] == 'Bitcoin'][val], m*df.loc[df['crypto_name'] == 'Bitcoin'][val]+c, color = 'red')

    plt.xlabel(val)
    plt.ylabel('close')
    plt.title(f'Scatter plot between {val} and close ')
plt.subplots_adjust(left=0.1,
                    bottom=0.08, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.1, 
                    hspace=0.4)

#boxplot to check outliers with whisker_length(whis) of 1.5(default value)
plt.figure(num="Box plot")
for index, val in enumerate(visualize_cols):
    plt.subplot(3,2,index+1)
    plt.boxplot(pd.array(df.loc[df['crypto_name'] == 'Bitcoin'][val]), vert = False)
    plt.title(f'Box plot of {val} ')

plt.subplots_adjust(left=0.1,
                    bottom=0.08, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.1, 
                    hspace=0.4)
plt.show()

#checking null values in Bitcoin dataset
print(df.loc[df['crypto_name'] == 'Bitcoin'].describe(include='all'))
print(df.loc[df['crypto_name'] == 'Bitcoin'].isnull().sum())

df['open_close'] = df['open'] - df['close']
df['low_high'] = df['low'] - df['high']
df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)

df.tail()