import numpy as np
import matplotlib.pyplot as plt

# import tensorflow as tf
# import tensorflow.contrib.rnn as rnn
# import tensorflow.contrib.metrics as metrics

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import optimizers
from keras import backend as KBend

from sklearn.metrics import mean_squared_error
import pandas
import math

np.random.seed(7)

## x(t) = phi0 + phi1*x(t-1) + a(t)
def ar1(phi0, phi1, length=10):
    x = [1]
    for i in range(0,length-1):
        x.append(phi0 + phi1*x[i] + np.random.normal(0,1,1)[0])
    return x

################################################################################################################################################
################################################################################################################################################

# xsize = 400
# x = ar1(0.9,0.9,xsize)
# x = (x-np.ones(len(x))*min(x))/(max(x)-min(x))
# xtrain = x[0:xsize/2]
# xtest = x[xsize/2:xsize]

################################################################################################################################################
################################################################################################################################################

data = pandas.read_csv('data/daily-returns.csv', sep=r"\s+", header=None)
data = data.values
data = np.reshape(data, data.shape[0])
data = (data-np.ones(len(data))*min(data))/(max(data)-min(data))
data = data/max(data)
xtrain = data[0:len(data)/2]
xtest = data[len(data)/2:len(data)]
# print len(data)

################################################################################################################################################
################################################################################################################################################

def create_dataset_for_lstm(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append([dataset[i + look_back]])
	return np.array(dataX), np.array(dataY)

rmse_test = []
look_backs = range(1,40,2)
for look_back in look_backs:
    trainX, trainY = create_dataset_for_lstm(xtrain, look_back)
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    # trainY = np.reshape(trainY, (trainY.shape[0], 1, trainY.shape[1]))
    testX, testY = create_dataset_for_lstm(xtest, look_back)
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    ######## create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(8, input_shape=(1, look_back)))
    model.add(Dense(1))

    sgd = optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
    rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    adagrad = optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
    def root_mean_squared_error(y_true, y_pred):
        return KBend.sqrt(KBend.mean(KBend.square(y_pred - y_true), axis=-1))
    # model.compile(loss=root_mean_squared_error, optimizer=sgd, metrics=['accuracy'])
    model.compile(loss=root_mean_squared_error, optimizer=adagrad, metrics=['accuracy'])
    history = model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)


    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
    # print('Train Score: %.6f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY, testPredict))
    # print('Test Score: %.6f RMSE' % (testScore))

    rmse_test.append(testScore)


plt.plot(look_backs, rmse_test, label='test RMSE vs lookback')
plt.xlabel('lookback')
plt.ylabel('RMSE')
plt.show()
