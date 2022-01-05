# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 10:20:43 2021

@author: zhang
"""

from matplotlib import pyplot
from pandas import read_csv
from sklearn import preprocessing
from math   import sqrt
from numpy  import concatenate
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_squared_error , mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM , GRU , SimpleRNN
from keras.layers import Dense , Dropout , Activation
import numpy as np
from sklearn import metrics
#import os

def series_to_supervised(data,n_in=1,n_out=1,dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    #n_vars=data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    
    for i in range(n_in,0,-1):
        cols.append(df)
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0,n_out):
        cols.append(df.shift(-i))
        if i==0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg=concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

dataset = read_csv('0827_-10oC_zhire_trainedcon.csv' , header = 0 , index_col = 0)

values = dataset.values

encoder=preprocessing.LabelEncoder()

values = values.astype('float32')
scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(values)

reframed = series_to_supervised(scaled,1,1)

reframed.drop(reframed.columns[[6,7,8,9,10,11,12]],axis=1,inplace=True)#此处出现的报错是舍掉的列数不对造成的

print(reframed.head())

values = reframed.values

times=10
dropout=0 
results=[]
rmse_data=[]
mae_data=[]
r2_data=[]
acc_data=[]

rmse_data2=[]
mae_data2=[]
r2_data2=[]
acc_data2=[]

rmse_data3=[]
mae_data3=[]
r2_data3=[]
acc_data3=[]

for i in range(0,times):
    n_train_datas = 5215

    train = values[:n_train_datas,:]
    test = values[n_train_datas:n_train_datas+1311,:]
    
    train_X, train_y = train[:,:-1], train[:, -1]
    test_X, test_y = test[:,:-1], test[:, -1]
    
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    
    model2 = Sequential()

    l = 2

    model2.add(GRU(l,input_shape=(train_X.shape[1],train_X.shape[2])))

    print(train_X.shape[2])
    model2.add(Dropout( dropout ))
    model2.add(Dense(1))
    model2.compile(loss='mse', optimizer='adam')
    model2.summary()
      
    history2=model2.fit(train_X, train_y, epochs=200, batch_size=30, validation_data=(test_X , test_y), verbose=1  )
    yhat2 = model2.predict(test_X)

    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

    inv_yhat2 = concatenate((test_X , yhat2), axis=1)
    inv_yhat2 = scaler.inverse_transform(inv_yhat2)
    inv_yhat2 = inv_yhat2[:, -1]
    inv_yhat2 = np.array(inv_yhat2)
    
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_X , test_y), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, -1]
    
    rmse2 = sqrt(mean_squared_error(inv_y, inv_yhat2))
    
    R2_2 = metrics.r2_score(inv_y, inv_yhat2)
    mae_2 = mean_absolute_error(inv_y, inv_yhat2)

    pyplot.plot(inv_y,label='inv_y')
    pyplot.plot(inv_yhat2,label='inv_yhat')
    pyplot.legend()
    pyplot.show()
    acc_GRU = 1-np.mean(abs(inv_yhat2 - inv_y)/inv_y)
       
    results=np.append(results,'rmse')
    results=np.append(results,rmse2)
    
    results=np.append(results,'R2')
    results=np.append(results,R2_2)
    
    results=np.append(results,'mae')
    results=np.append(results,mae_2)
    
    results=np.append(results,'acc')
    results=np.append(results,acc_GRU)
   
    rmse_data2=np.append(rmse_data2,rmse2)
    r2_data2=np.append(r2_data2,R2_2)
    mae_data2=np.append(mae_data2,mae_2)
    acc_data2=np.append(acc_data2,acc_GRU)
    
results=results.reshape(times,4,2)
print(results)