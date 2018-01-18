import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import math

np.random.seed(7)

## x(t) = c + A*x(t-1) + a(t)
def var1(c, A, length=10):
    dim = len(c)
    x = [np.ones(dim)]
    for i in range(0,length-1):
        x.append(c + np.multiply(A,x[i]) + np.random.normal(0,1,dim))
    return np.array(x)

dim = 9 # dimension of x
c = np.random.normal(0,1,dim)
A = np.ones([dim, dim])*0.9
x = var1(c,0.9,2)

