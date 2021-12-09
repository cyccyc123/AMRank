import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

class DocGan(keras.Model):
    def __init__(self, feature_dim=45):
        super(DocGan, self).__init__()
        self.checkpoint_file = 'tmp/gan'
        self.feature_dim = feature_dim

        self.fc = Dense(25, activation="relu",
                       kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.0001),
                       bias_initializer=tf.constant_initializer(0.0),
                       name="fc")

        self.gan = Dense(1,
                         kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.0001),
                         bias_initializer=tf.constant_initializer(0.0),
                         name="gan")

    def call(self, state):
        f = self.fc(state)
        prob = self.gan(f)
        return prob






