#set time as a index
type(data.index[0])

#set time as a index
data=data.set_index('time')
data.head(2)

# set the time and indet according to the data
data.index=pd.to_datetime(data.index)
data.head(2)