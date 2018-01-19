import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import optimizers
from keras import backend as KBend

from sklearn.metrics import mean_squared_error
import pandas
import math

np.random.seed(7)

def arch1(phi0, phi1, alpha0, alpha1, length=10):
	x = [0]
	sigma = 1
	e = [np.random.normal(0,sigma,1)[0]]
	x.append(phi0 + phi1*x[0] + e[0])

	for i in range(1,length-1):
		sigma_sq = alpha0 + alpha1*e[i-1]*e[i-1]
		e.append(np.random.normal(0,np.sqrt(sigma_sq),1)[0])
		x.append(phi0 + phi1*x[i] + e[i])
	return x

xsize = 1000
x = np.array(arch1(0.9,0.9,0.9,0.9,xsize))

xtrain = x[0:xsize/2]
xtest = x[xsize/2:xsize]

plt.plot(np.array(range(x.shape[0])), x, label='ARCH(1) data')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.show()


############# dataset preprocessing ###############
def create_dataset_for_lstm(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append([dataset[i + look_back]])
	return np.array(dataX), np.array(dataY)

look_back = 10
trainX, trainY = create_dataset_for_lstm(xtrain, look_back)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX, testY = create_dataset_for_lstm(xtest, look_back)
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


######## create and fit the LSTM network ###########
model = Sequential()
model.add(LSTM(8, input_shape=(1, look_back)))
model.add(Dense(1))

sgd = optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
adagrad = optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
def root_mean_squared_error(y_true, y_pred):
	return KBend.sqrt(KBend.mean(KBend.square(y_pred - y_true), axis=-1))
# model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy']) # no learning
# model.compile(loss='mean_squared_error', optimizer=adagrad, metrics=['accuracy']) # learned only below mean
model.compile(loss='mean_squared_error', optimizer=rmsprop, metrics=['accuracy']) # learned best
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy']) # learned
history = model.fit(trainX, trainY, epochs=20, batch_size=1, verbose=2)


trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.6f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.6f RMSE' % (testScore))

plt.plot(np.array(range(trainX.shape[0])), np.reshape(trainPredict, trainPredict.shape[0]), label='LSTM on train')
plt.plot(np.array(range(trainX.shape[0])), np.reshape(trainY, trainY.shape[0]), label='ARCH(1)')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend()
plt.show()

plt.plot(np.array(range(testPredict.shape[0], testPredict.shape[0]+testPredict.shape[0])), np.reshape(testPredict, testPredict.shape[0]), label='LSTM on test')
plt.plot(np.array(range(testPredict.shape[0], testPredict.shape[0]+testPredict.shape[0])), np.reshape(testY, testY.shape[0]), label='ARCH(1)')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend()
plt.show()


## with rmsprop
# Train Score: 1.990961 RMSE
# Test Score: 2.675569 RMSE
