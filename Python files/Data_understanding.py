import pandas as pd              #Brings in the Pandas library and aliases it as 'pd' for data manipulation.




df = pd.read_csv('dataset.csv')  #Reads data from 'dataset.csv' and stores it in a DataFrame called 'df'.
df.head()                        #Shows the initial rows of the DataFrame, offering a quick view of the dataset.
df.shape                         #Provides the number of rows and columns in the DataFrame, indicating its size.
df.describe()                    #Presents summary statistics for numerical columns, revealing central tendencies and data spread.

