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
from keras.layers import Dense , Dropout , Activation
import numpy as np
import keras
from sklearn.svm import SVR
from sklearn import metrics

#import os

         
loss_ = 'mse'
optimizer_ = 'adam'
"""
wind风速档位
tep温度档位
ET环境温度
EH环境湿度
RH相对湿度
WS风速
SOLAR照度
OUT_AVE出风口平均温度
FEET_AVE脚部平均温度
FEET_OUT_AVE脚部出风口平均温度
MID_OUT中出风口
RIGHT_OUT右出风口
LEFT_OUT左出风口
LEFT_FEET_OUT左脚出风口
LEFT_FEET主驾脚部
INNER_TEP室内温度
LEFT_HEAD主驾头部

"""
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

values = values.astype('float32')
values = np.array(values)
scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(values)


reframed = series_to_supervised(scaled,1,1)


reframed.drop(reframed.columns[[6,7,8,9,10,11,12]],axis=1,inplace=True)#此处出现的报错是舍掉的列数不对造成的

print(reframed.head())

values = reframed.values

times=10
dropout=0
results=[]
results_SVR=[]
rmse_data5=[]
rmse_data4=[]
mae_data4=[]
r2_data4=[]
acc_data4=[]
rmse_data5=[]
mae_data5=[]
r2_data5=[]
acc_data5=[]



ANN_data_out=[]
SVR_data_out=[]

for i in range(0,times):
    n_train_datas = 5215

    train = values[:n_train_datas,:]
    test = values[n_train_datas:n_train_datas+1311,:]
    
    train_X, train_y = train[:,:-1], train[:, -1]
    test_X, test_y = test[:,:-1], test[:, -1]

    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    

    model4 = Sequential()
#    hidden_layer_1 = int(i/3)+5
    hidden_layer_1 = 14
#    hidden_layer_2 = int(i/3)+1
    hidden_layer_2 = 14

    model4.add(Dense(output_dim=hidden_layer_1, kernel_initializer='random_normal', input_dim=6, bias_initializer=keras.initializers.RandomNormal(mean = 0.3, stddev=0.05, seed=None)))
    model4.add(Activation('tanh'))
    #model4.add(Dense(5*(int(i/5)+1)*5, kernel_initializer='random_normal', bias_initializer=keras.initializers.RandomNormal(mean = 0.3, stddev=0.05, seed=None)))
    model4.add(Dense(hidden_layer_2, kernel_initializer='random_normal', bias_initializer=keras.initializers.RandomNormal(mean = 0.3, stddev=0.05, seed=None)))
#    ,kernel_regularizer=keras.regularizers.l2(0.001) L2正则化
    model4.add(Activation('tanh'))
    model4.add(Dropout(dropout))
    model4.add(Dense(output_dim=1, kernel_initializer='random_normal', bias_initializer=keras.initializers.RandomNormal(mean = 0.3, stddev=0.05, seed=None)))
    model4.add(Activation('tanh'))
    
    #model4.compile(loss='mae', optimizer='adam')
#    model4.compile(loss='mse', optimizer='adam') 
    model4.compile(loss=loss_, optimizer=optimizer_) 
    model4.summary()

    
#    history4=model4.fit(train_X, train_y, epochs=200, batch_size=30, validation_data=(test_X , test_y), callbacks=[reduce_lr])
    history4=model4.fit(train_X, train_y, epochs=300, batch_size=30, validation_data=(test_X , test_y))

    yhat4 = model4.predict(test_X)
    yhat4 = yhat4.reshape(len(yhat4),1)


    inv_yhat4 = concatenate((test_X , yhat4), axis=1)
    inv_yhat4 = scaler.inverse_transform(inv_yhat4)
    inv_yhat4 = inv_yhat4[:, -1]
    inv_yhat4 = np.array(inv_yhat4)
    
    model5 = SVR(kernel='rbf', C = 0.285, gamma = 0.42, epsilon = 0.001)   #径向基核函数初始化的SVR
    model5.fit(train_X, train_y)
    yhat5 = model5.predict(test_X)
    yhat5 = yhat5.reshape(len(yhat5),1)
    
    inv_yhat5 = concatenate((test_X , yhat5), axis=1)
    inv_yhat5 = scaler.inverse_transform(inv_yhat5)
    inv_yhat5 = inv_yhat5[:, -1]
    inv_yhat5 = np.array(inv_yhat5)
    
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_X , test_y), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, -1]
    
    rmse4 = sqrt(mean_squared_error(inv_y, inv_yhat4))
    R2_4 = metrics.r2_score(inv_y, inv_yhat4)
    mae_4 = mean_absolute_error(inv_y, inv_yhat4)
    acc_ANN = 1-np.mean(abs(inv_yhat4 - inv_y)/inv_y)
    
    ANN_data_out=np.append(ANN_data_out,inv_yhat4)
    SVR_data_out=np.append(SVR_data_out,inv_yhat5)
    
    results=np.append(results,'rmse')
    results=np.append(results,rmse4)
    results=np.append(results,'R2')
    results=np.append(results,R2_4)
    results=np.append(results,'mae')
    results=np.append(results,mae_4)
    results=np.append(results,'acc')
    results=np.append(results,acc_ANN)
    
    rmse_data4=np.append(rmse_data4,rmse4)
    r2_data4=np.append(r2_data4,R2_4)
    mae_data4=np.append(mae_data4,mae_4)
    acc_data4=np.append(acc_data4,acc_ANN)
    

    
    rmse_5 = sqrt(mean_squared_error(inv_y, inv_yhat5))
    R2_5 = metrics.r2_score(inv_y, inv_yhat5)
    mae_5 = mean_absolute_error(inv_y, inv_yhat5)
    acc_5 = 1-np.mean(abs(inv_yhat5 - inv_y)/inv_y)
       
    results_SVR=np.append(results_SVR,'rmse')
    results_SVR=np.append(results_SVR,rmse_5)
    results_SVR=np.append(results_SVR,'R2')
    results_SVR=np.append(results_SVR,R2_5)
    results_SVR=np.append(results_SVR,'mae')
    results_SVR=np.append(results_SVR,mae_5)
    results_SVR=np.append(results_SVR,'acc')
    results_SVR=np.append(results_SVR,acc_5)
    rmse_data5=np.append(rmse_data5,rmse_5)
    r2_data5=np.append(r2_data5,R2_5)
    mae_data5=np.append(mae_data5,mae_5)
    acc_data5=np.append(acc_data5,acc_5)
    
rmse_ave = sum(rmse_data4)/len(rmse_data4)
r2_ave = sum(r2_data4)/len(r2_data4)
mae_ave = sum(mae_data4)/len(mae_data4)
acc_ave = sum(acc_data4)/len(acc_data4)   
print('rmse:%.10f,r2:%.10f,mae:%.10f,acc:%.10f' % (rmse_ave,r2_ave,mae_ave,acc_ave))
  
results_SVR=results_SVR.reshape(times,4,2)
results=results.reshape(times,4,2)

print('results_ANN:')
print(results)
print('----------------------------------------')
print('results_SVR:')
print(results_SVR)

ANN_data_out = ANN_data_out.reshape(-1,len(test_y))
SVR_data_out = SVR_data_out.reshape(-1,len(test_y))
ANN_data_out = np.array(ANN_data_out)
SVR_data_out = np.array(SVR_data_out)
ANN_data_out = ANN_data_out.T
SVR_data_out = SVR_data_out.T

for j in range(0,times):
    pyplot.plot(inv_y, label='true')
    pyplot.plot(SVR_data_out[:,j],label='SVR')
    pyplot.plot(ANN_data_out[:,j],label='ANN')
    pyplot.legend()
    pyplot.show()
print('----------------------------------------')

print(loss_,optimizer_)
print('rmse:%.10f,r2:%.10f,mae:%.10f,acc:%.10f' % (rmse_ave,r2_ave,mae_ave,acc_ave))