import pandas as pd              #Brings in the Pandas library and aliases it as 'pd' for data manipulation.
import matplotlib.pyplot as plt  #Importing Matplotlib for plotting the data
import seaborn as sb             #Import seaborn for plotting graphs or make visualizations


df = pd.read_csv('dataset.csv')  #Reads data from 'dataset.csv' and stores it in a DataFrame called 'df'.
df.head()                        #Shows the initial rows of the DataFrame, offering a quick view of the dataset.
df.shape                         #Provides the number of rows and columns in the DataFrame, indicating its size.
df.describe()                    #Presents summary statistics for numerical columns, revealing central tendencies and data spread.

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
