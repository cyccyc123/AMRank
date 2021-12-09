import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, GRU

class SeqGan(keras.Model):
    def __init__(self, feature_dim=45, heads=9):
        super(SeqGan, self).__init__()
        self.checkpoint_file = 'tmp/gan2'
        self.feature_dim = feature_dim
        self.heads = heads
        self.key = Dense(feature_dim//heads)
        self.value = Dense(feature_dim//heads)
        self.query = Dense(feature_dim//heads)
        self.key_ = Dense(feature_dim % heads)
        self.value_ = Dense(feature_dim % heads)
        self.query_ = Dense(feature_dim % heads)
        self.ln0 = tf.keras.layers.LayerNormalization()
        self.ln1 = tf.keras.layers.LayerNormalization()

        self.gru = GRU(25, activation="tanh",
                       kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.0001),
                       bias_initializer=tf.constant_initializer(0.0),
                       name="gru")

        self.gan = Dense(1,
                         kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.0001),
                         bias_initializer=tf.constant_initializer(0.0),
                         name="gan")

    def call(self, state):
        batch_size = state.shape[0]
        sequence_length = state.shape[1]
        feature_dim = self.feature_dim
        heads = self.heads
        if feature_dim % heads == 0:
            x_mh = tf.reshape(state, (batch_size, sequence_length, heads, feature_dim//heads))
            x_mh = tf.transpose(x_mh, (0, 2, 1, 3))
            key = self.key(x_mh)
            value = self.value(x_mh)
            query = self.query(x_mh)
            atten = tf.matmul(query, key, transpose_b=True)
            atten = atten / tf.sqrt(tf.cast(feature_dim//heads, tf.float32))
            atten = tf.nn.softmax(atten)
            y = tf.matmul(atten, value)
            y = tf.transpose(y, [0, 2, 1, 3])
            y = tf.reshape(y, (batch_size, sequence_length, feature_dim))
        else:
            state_ = np.array(state[:, :, feature_dim//heads*heads:])
            state = np.array(state[:, :, :feature_dim//heads*heads])
            x_mh = tf.reshape(state, (batch_size, sequence_length, heads, feature_dim//heads))
            x_mh = tf.transpose(x_mh, (0, 2, 1, 3))
            key = self.key(x_mh)
            value = self.value(x_mh)
            query = self.query(x_mh)
            atten = tf.matmul(query, key, transpose_b=True)
            atten = atten / tf.sqrt(tf.cast(feature_dim//heads, tf.float32))
            atten = tf.nn.softmax(atten)
            y = tf.matmul(atten, value)
            y = tf.transpose(y, [0, 2, 1, 3])
            y = tf.reshape(y, (batch_size, sequence_length, feature_dim//heads*heads))
            x_mh_ = tf.reshape(state_, (batch_size, sequence_length, 1, feature_dim % heads))
            x_mh_ = tf.transpose(x_mh_, (0, 2, 1, 3))
            key_ = self.key_(x_mh_)
            value_ = self.value_(x_mh_)
            query_ = self.query_(x_mh_)
            atten_ = tf.matmul(query_, key_, transpose_b=True)
            atten_ = atten_ / tf.sqrt(tf.cast(feature_dim % heads, tf.float32))
            atten_ = tf.nn.softmax(atten_)
            y_ = tf.matmul(atten_, value_)
            y_ = tf.transpose(y_, [0, 2, 1, 3])
            y_ = tf.reshape(y_, (batch_size, sequence_length, feature_dim % heads))
            y = tf.concat([y, y_], axis=2)
        y = self.ln0(state + y)
        g = self.gru(y)
        prob = self.gan(g)
        return prob






