import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
# from tensorflow.python.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.layers import Layer

# W1 = np.random.uniform(0, 1, size=(18, 50))
# W2 = np.random.uniform(0, 1, size=(4, 50))
# x = np.random.uniform(0, 1, size=(100, 50))
# M = np.tanh(np.matmul(W1, x.T))
# alpha = np.exp(M) / (np.sum(np.exp(M), axis=1).reshape((18, -1)))  # softmax
# alpha = np.expand_dims(alpha, axis=-1)
# r = np.sum(alpha * x, axis=1)
# y = np.matmul(r, W2.T)
# y = np.exp(y) / np.sum(np.exp(y), axis=1).reshape((18, -1))  # softmax
# print(y.shape)
W1 = K.cast(np.random.uniform(0, 1, size=(50, 18)), K.floatx())
W2 = K.cast(np.random.uniform(0, 1, size=(50, 4)), K.floatx())
x = K.cast(np.random.uniform(0, 1, size=(32, 100, 50)), K.floatx())
M = K.tanh(K.reshape(K.dot(K.reshape(x, (-1, 50)), W1), (-1, 100)))
alpha = K.exp(M)
alpha /= K.sum(alpha, axis=-1, keepdims=True)
alpha = K.expand_dims(K.reshape(alpha, (-1, 18, 100)), axis=-1)
r = alpha * x
print(r)





