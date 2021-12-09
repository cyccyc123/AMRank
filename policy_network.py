import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

class PolicyNetwork(keras.Model):
    def __init__(self, feature_dim=45):
        super(PolicyNetwork, self).__init__()
        self.checkpoint_file = 'tmp/policy'
        self.feature_dim = feature_dim
        self.actor = Dense(1,
                         kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.0001),
                         bias_initializer=tf.constant_initializer(0.1),
                         name="actor")

    def call(self, state):
        score = self.actor(state)
        return score






