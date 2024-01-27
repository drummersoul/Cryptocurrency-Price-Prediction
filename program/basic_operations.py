import pandas as pd 

df = pd.read_csv(r'C:\Users\kiran\OneDrive\Desktop\crypto prediction\Cryptocurrency-Price-Prediction\excel\dataset.csv')

print(df)

# print(df) """display all the data set"""
# print(df.head(4)) """display top 4 rows"""
# print(df.tail(4)) """ return last 4 rows"""
# print(df['open']) """ return open cloum details"""
# print(df[['open','high','low']]) """return open, high, low column details"""
# print(df.head(4)[['open', 'high', 'low']]) """disply top 4 open, high, low column details"""

# print(df.iloc[2]) """display index 2 row details"""
# print(df.iloc[0:4]) """display data from 0 index to 3rd index """
# print(df.iloc[2,2]) """display a perticular data with in 2nd index row and 2nd index cloumn"""
# print(df.iloc[0:4,0:4]) """display data with in 0 to 3rd index row and o to 3rd index cloumn"""

# print(df.loc[df['crypto_name'] == 'Bitcoin']) """filter data based on column name"""
# print(df.loc[(df['crypto_name'] == 'Bitcoin') & (df['close'] > 115)]) """filter data with and  condition"""
# print(df.loc[(df['crypto_name'] == 'Bitcoin') | (df['close'] > 115)]) """filter data with or condition"""

# print(df[['open', 'high', 'low', 'close', 'volume', 'marketCap']].loc[df['crypto_name'] == 'Bitcoin'].sum()) """ sum all the cloumn values"""



#program for calculating avg for all crypto

crypto_name = df['crypto_name'].unique() #return unquie values in the respective row
print(crypto_name)

final_avg_df_ls = pd.DataFrame()

for crypto in crypto_name:
    avg_df = df[['open', 'high', 'low', 'close', 'volume', 'marketCap']].loc[df['crypto_name'] == 'Bitcoin'].mean() #calculate mean or avg
    final_avg_df_ls = pd.concat([final_avg_df_ls, pd.DataFrame({"crypto_name" :[crypto], "avg_open" : [avg_df['open']], "avg_high" : [avg_df['high']],
                         "avg_close" : [avg_df['close']], "avg_volume" : [avg_df['volume']], "avg_marketCap" : [avg_df['marketCap']]})]
                         , ignore_index=True)

print(final_avg_df_ls.sort_values(['crypto_name'])) #sort the data
