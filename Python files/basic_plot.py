import pandas as pd
import matplotlib.pyplot as py


#get data from file...
df = pd.read_csv(r'C:\Users\kiran\OneDrive\Desktop\crypto prediction\crypto_week5\Cryptocurrency-Price-Prediction\Excel DB\Crypto_data_info.csv')
#print(df) print df
#print(df.to_string()) print df in stringformat

print(df.size) #return no.of rows

# print(df.info())
# print(df.__len__)

# print(df.loc[df['crypto_name'] == 'Bitcoin'].__len__)

#
#=================ploting graphs=======================
#

bc_df = df.loc[df['crypto_name'] == 'Bitcoin']
print(bc_df)
# print(bc_df.shape)

# py.subplot(2,2,1)
# py.plot(bc_df.head(10)['date'], bc_df.head(10)['open'])
# py.xlabel('date')
# py.ylabel('open')

# py.subplot(2,2,2)
# py.plot(bc_df.head(10)['date'], bc_df.head(10)['close'])
# py.xlabel('date')
# py.ylabel('close')

# py.subplot(2,2,3)
# py.plot(bc_df.head(10)['date'], bc_df.head(10)['low'])
# py.xlabel('date')
# py.ylabel('low')

# py.subplot(2,2,4)
# py.plot(bc_df.head(10)['date'], bc_df.head(10)['high'])
# py.xlabel('date')
# py.ylabel('high')

# py.show()

#
#============================Data cleaning======================
#

print(bc_df.info())
print(bc_df.describe())

# py.plot(bc_df['close']) #given y-axis, system will take default x-axis from 0
# print(bc_df['close'].max())
# print(bc_df['close'].min())
# py.show()

# print(pd.to_datetime(df['date']).dt.year) #to get year in a date objects

print(bc_df.isnull().sum())