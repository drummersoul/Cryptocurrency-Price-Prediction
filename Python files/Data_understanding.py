import pandas as pd              #Brings in the Pandas library and aliases it as 'pd' for data manipulation.
import matplotlib.pyplot as plt  #Importing Matplotlib for plotting the data
import seaborn as sb             #Import seaborn for plotting graphs or make visualizations

df = pd.read_csv(r'C:\Users\kiran\OneDrive\Desktop\crypto prediction\crypto_week5\Cryptocurrency-Price-Prediction\Excel DB\Crypto_data_info.csv')  #Reads data from 'dataset.csv' and stores it in a DataFrame called 'df'.
df.head()                        #Shows the initial rows of the DataFrame, offering a quick view of the dataset.
df.shape                         #Provides the number of rows and columns in the DataFrame, indicating its size.
df.describe()                    #Presents summary statistics for numerical columns, revealing central tendencies and data spread.

print(df)

# Set the figure size to 10x5 inches.
plt.figure(figsize=(10,5))

# Plot the 'close' column from the DataFrame.
plt.plot(df['close'])  

# Set the title of the plot with a fontsize of 15.
plt.title('Bitcoin Close price.', fontsize=15)

# Label the y-axis as 'Price in dollars.'
plt.ylabel('Price in dollars.')  

#display the plot
plt.show()

print(df.isnull().sum())

#We create a list named features that will contain the features or columns that we want to explore
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