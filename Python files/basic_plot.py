import pandas as pd
import matplotlib.pyplot as plt


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

#boxplot to check outliers
plt.figure()
for index, val in enumerate(['open', 'high', 'low', 'close', 'volume', 'marketCap']):
    plt.subplot(3,2,index+1)
    plt.boxplot(pd.array(df.loc[df['crypto_name'] == 'Bitcoin'][val]), vert=0)
    plt.title(f'Box plot of {val} ')
plt.subplots_adjust(left=0.1,
                    bottom=0.08, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.1, 
                    hspace=0.4)

# trying plot for different whisker lenght
plt.figure()
for index, val in enumerate(['open', 'high', 'low', 'close', 'volume', 'marketCap']):
    plt.subplot(3,2,index+1)
    plt.boxplot(pd.array(df.loc[df['crypto_name'] == 'Bitcoin'][val]), vert=0, whis=3.5)
    plt.title(f'Box plot of {val} ')
plt.subplots_adjust(left=0.1,
                    bottom=0.08, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.1, 
                    hspace=0.4)
plt.show()