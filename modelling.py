# coding=utf-8
import tensorflow as tf
from tensorflow import keras

class WindowLayer(keras.layers.Layer):
    """
    """
    def __init__(self, num_mixtures, sequence, num_letters):
        self.c = sequence # one-hot encoding sequence of characters -> [batch_size, length, num_letters]
        self.num_mixtures = num_mixtures # K
        self.num_letters = num_letters
        seq_len = tf.shape(sequence)[1]
        self.u = tf.range(0., tf.cast(seq_len, dtype=tf.float32)) # [U]
        super(WindowLayer, self).__init__()
  
    def get_phi(self, alpha, beta, kappa, check_end=False):
        u = [tf.cast(tf.shape(self.u)[0], tf.float32)] if check_end is True else self.u
        alpha = tf.expand_dims(alpha, axis=2) # [batch_size, num_mixtures, 1] 
        beta = tf.expand_dims(beta, axis=2) # [batch_size, num_mixtures, 1] 
        kappa = tf.expand_dims(kappa, axis=2) # [batch_size, num_mixtures, 1] 
        phi =  alpha * tf.exp(-beta * tf.square(kappa - self.u)) # [batch, num_mixtures, U]
        return tf.reduce_sum(phi, axis=1, keepdims=True) # [batch_size, 1, U]

    def call(self, inputs, states):
        prev_kappa = states[0]
        alpha = keras.layers.Dense(self.num_mixtures, activation=tf.exp)(inputs) # [batch_size, num_mixtures] 
        beta = keras.layers.Dense(self.num_mixtures, activation=tf.exp)(inputs) # [batch_size, num_mixtures]
        kappa = prev_kappa + keras.layers.Dense(self.num_mixtures, activation=tf.exp)(inputs) # [batch_size, num_mixtures]
        phi = self.get_phi(alpha, beta, kappa) # [batch_size, 1, U]
#         finish = tf.squeeze(self.get_phi(alpha, beta, kappa, check_end=True), axis=1) > tf.reduce_max(phi, axis=1) # [batch_size]
        return tf.squeeze(tf.matmul(phi, self.c), axis=1), \
                kappa
    @property
    def state_size(self):
        return [self.num_mixtures]
      
    @property
    def output_size(self):
        return [self.num_letters, self.num_mixtures, 1]
          

class MixtureLayer(keras.layers.Layer):
    def __init__(self, input_size, num_mixtures, bias = 0):
        self.input_size = input_size
        self.num_mixtures = num_mixtures
        self.bias = bias
        super(MixtureLayer, self).__init__()
    
    def call(self, inputs):
        e = keras.layers.Dense(1, activation=tf.sigmoid)(inputs)
        pi = keras.layers.Dense(self.num_mixtures)(inputs)
        mu1 = keras.layers.Dense(self.num_mixtures)(inputs)
        mu2 = keras.layers.Dense(self.num_mixtures)(inputs)
        std1 = keras.layers.Dense(self.num_mixtures, activation=tf.exp)(inputs)
        std2 = keras.layers.Dense(self.num_mixtures, activation=tf.exp)(inputs)
        rho = keras.layers.Dense(self.num_mixtures, activation=tf.tanh)(inputs)
        
        return keras.layers.concatenate([e, tf.nn.softmax(pi * (1. + self.bias)), mu1, mu2, std1, std2, rho])

def create_model(num_letters, num_mixtures=10, num_layers=3, units=400):
    coordinates = keras.Input(shape=(None, 3))
    sequence = keras.Input(shape=(None, num_letters))
    
    lstms = [keras.layers.LSTM(units, return_sequences=True) for _ in range(num_layers)]
    window = WindowLayer(num_mixtures=num_mixtures, sequence=sequence, num_letters=num_letters)
    mixture = MixtureLayer(input_size=units, num_mixtures=num_mixtures)

    first_output = lstms[0](coordinates)
    window_output = keras.layers.RNN(window, return_sequences=True)(first_output)
    prev_output = []
    for i in range(1, num_layers):
        current_input = keras.layers.concatenate([coordinates, window_output] + prev_output, axis=2)
        output = lstms[i](current_input)
        prev_output = [output]
        
    final_output = mixture(output)
    model = keras.Model(inputs=[coordinates, sequence], outputs=final_output)
    return model
    