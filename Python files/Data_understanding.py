import pandas as pd              #Brings in the Pandas library and aliases it as 'pd' for data manipulation.
import matplotlib.pyplot as plt  #Importing Matplotlib for plotting the data


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

