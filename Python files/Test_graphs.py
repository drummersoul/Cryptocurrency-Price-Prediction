from graphs import Graphs
import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\kiran\OneDrive\Desktop\crypto prediction\crypto_week5\Cryptocurrency-Price-Prediction\Excel DB\Crypto_data_info.csv')

print(type(df.loc[df['crypto_name'] == 'Bitcoin']))

grp = Graphs()


print(type('close'))
grp.boxPlotWithSubplot(df.loc[df['crypto_name'] == 'Bitcoin'])