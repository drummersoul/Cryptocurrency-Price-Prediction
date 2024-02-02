import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\kiran\OneDrive\Desktop\crypto prediction\Cryptocurrency-Price-Prediction\excel\dataset.csv')

print(df.isnull().values.any()) #check is there any null values in the dataset

#scatter plot for close & date for bitcoin
for crypto in ['Bitcoin', 'Litecoin']:
    df1 = df.tail(int(df.size/2))[['close', 'date']].loc[df['crypto_name'] == crypto]
    df1.plot.scatter(x = 'date', y= 'close')
    plt.title(f"{crypto} relation b/w close and date")
plt.show()