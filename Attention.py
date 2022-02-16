import tensorflow as tf
from tensorflow.keras import backend as K
# from tensorflow.python.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.layers import Layer


class AttOutput(Layer):

    def __init__(self, step_dim, class_num,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.class_num = class_num
        self.features_dim = 0

        super(AttOutput, self).__init__(**kwargs)

    def build(self, input_shape):

        assert len(input_shape) == 3

        self.features_dim = input_shape[-1]

        self.W1 = self.add_weight(shape=(input_shape[-1],),  # 50
                                  initializer=self.init,
                                  name='{}_W1'.format(self.name),
                                  regularizer=self.W_regularizer,
                                  constraint=self.W_constraint)

        self.W2 = self.add_weight(shape=(input_shape[-1], self.class_num),  # 50*4
                                  initializer=self.init,
                                  name='{}_W2'.format(self.name),
                                  regularizer=self.W_regularizer,
                                  constraint=self.W_constraint)


        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        # 这里应该是 step_dim是我们指定的参数，它等于input_shape[1],也就是rnn的timesteps
        step_dim = self.step_dim

        # 输入和参数分别reshape再点乘后，tensor.shape变成了(batch_size*timesteps, 1),之后每个batch要分开进行归一化
        # 所以应该有 e = K.reshape(..., (-1, timesteps))
        e = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W1, (features_dim, 1))),
                      (-1, step_dim))  # e = K.dot(x, self.W)
        if self.bias:
            e += self.b
        # RNN一般默认激活函数为tanh, 对attention来说激活函数差别不打，因为要做softmax
        e = K.tanh(e)

        a = K.exp(e)
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # 如果前面的层有mask，那么后面这些被mask掉的timestep肯定是不能参与计算输出的，也就是将他们的attention权重设为0
            # cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # cast是做类型转换，keras计算时会检查类型，可能是因为用gpu的原因
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        # a = K.expand_dims(a, axis=-1) , axis默认为-1， 表示在最后扩充一个维度。
        # 比如shape = (3,)变成 (3, 1)
        a = K.expand_dims(a)
        # 此时a.shape = (batch_size, timesteps, 1), x.shape = (batch_size, timesteps, units)

        # a*x的shape为 (batch_size, timesteps, units), 每个timestep的输出向量已经乘上了该timestep的权重
        # a*x在axis=1上取和，返回值的shape为 (batch_size, 1, units)
        c = K.sum(a * x, axis=1)
        o = K.exp(K.dot(c, self.W2))
        o /= K.cast(K.sum(o, axis=-1, keepdims=True) + K.epsilon(), K.floatx())

        return o