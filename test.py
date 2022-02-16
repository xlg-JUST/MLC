import tensorflow as tf
from tensorflow.keras import Model, preprocessing
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
import numpy as np
from data_prepare import vocab, paded_sen_idx, test_labels
from Attention import AttOutput


sen_len = 100
wv_len = 50
vocab_size = len(vocab)


inputs = Input(shape=(sen_len,))
embedding = Embedding(input_dim=vocab_size, output_dim=wv_len, input_length=sen_len)(inputs)

hidden = Bidirectional(GRU(sen_len, return_sequences=True))(embedding)
attoutput1 = AttOutput(step_dim=hidden.shape[1], class_num=4)(hidden)
attoutput2 = AttOutput(step_dim=hidden.shape[1], class_num=4)(hidden)
attoutput3 = AttOutput(step_dim=hidden.shape[1], class_num=4)(hidden)
attoutput4 = AttOutput(step_dim=hidden.shape[1], class_num=4)(hidden)
attoutput5 = AttOutput(step_dim=hidden.shape[1], class_num=4)(hidden)
attoutput6 = AttOutput(step_dim=hidden.shape[1], class_num=4)(hidden)
attoutput7 = AttOutput(step_dim=hidden.shape[1], class_num=4)(hidden)
attoutput8 = AttOutput(step_dim=hidden.shape[1], class_num=4)(hidden)
attoutput9 = AttOutput(step_dim=hidden.shape[1], class_num=4)(hidden)
attoutput10 = AttOutput(step_dim=hidden.shape[1], class_num=4)(hidden)
attoutput11 = AttOutput(step_dim=hidden.shape[1], class_num=4)(hidden)
attoutput12 = AttOutput(step_dim=hidden.shape[1], class_num=4)(hidden)
attoutput13 = AttOutput(step_dim=hidden.shape[1], class_num=4)(hidden)
attoutput14 = AttOutput(step_dim=hidden.shape[1], class_num=4)(hidden)
attoutput15 = AttOutput(step_dim=hidden.shape[1], class_num=4)(hidden)
attoutput16 = AttOutput(step_dim=hidden.shape[1], class_num=4)(hidden)
attoutput17 = AttOutput(step_dim=hidden.shape[1], class_num=4)(hidden)
attoutput18 = AttOutput(step_dim=hidden.shape[1], class_num=4)(hidden)
outputs = K.reshape(concatenate([attoutput1, attoutput2, attoutput3, attoutput4, attoutput5, attoutput6, attoutput7, attoutput8,
                      attoutput9, attoutput10, attoutput11, attoutput12, attoutput13, attoutput14, attoutput15,
                      attoutput16, attoutput17, attoutput18], axis=-1), (-1, 18, 4))


def loss(y_true, y_pred):
    y_true = K.reshape(y_true, (-1,))
    y_true = tf.one_hot(y_true, depth=4)
    y_pred = K.reshape(y_pred, (-1, 4))
    loss = K.mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred))

    return loss


model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss=loss)
model.fit(x=paded_sen_idx, y=test_labels, epochs=5)




