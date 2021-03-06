import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import json, pickle

from modelling import create_model
from helpers import extract_pred_output, get_sentence_from_vector, plot_handwriting

epsilon = 1e-8

flags = tf.flags
FLAGS = flags.FLAGS

## Parameters
flags.DEFINE_string(
    "data_dir", "data/training",
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task."
)

flags.DEFINE_string(
    "output_dir", "output",
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task."
)

flags.DEFINE_integer(
    "window_mixtures", 10,
    "The number of mixtures ... "
)

flags.DEFINE_integer(
    "output_mixtures", 20,
    "The number of mixtures ... "
)

flags.DEFINE_integer(
    "batch_size", 64, "The size of the training batch"
)

flags.DEFINE_integer(
    "epochs", 30, "The number of epochs"
)

class DataGenerator(keras.utils.Sequence):
    "Generates data for Keras"
    def __init__(self, list_IDs, data, labels, batch_size=32, seq_len=256, shuffle=True):
        "Initialization"
        self.batch_size = batch_size
        self.data = data
        self.labels = labels
        self.list_IDs = list_IDs
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor((len(self.list_IDs) - 1)/ self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch (The extra index for getting the sequential y data)
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    @staticmethod
    def load_dataset():
        with open(os.path.join(FLAGS.data_dir, "data.pkl"), "rb") as pickle_file:
            dataset = pickle.load(pickle_file)
        data = [np.array(d) for d in dataset]
        with open(os.path.join(FLAGS.data_dir, "labels.pkl"), "rb") as pickle_file:
            labels = pickle.load(pickle_file)
        with open(os.path.join(FLAGS.data_dir, "translation.json")) as json_file:
            translation = json.load(json_file)
        return data, labels, translation

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        "Generates data containing batch_size samples # X : (n_samples, *dim, 3)"
        # get maximum sequence length before padding
        seq_len = -1
        str_len = -1
        for ID in list_IDs_temp:
            seq_len = max(self.data[ID].shape[0], seq_len)
            str_len = max(self.labels[ID].shape[0], str_len)
        
        # Initialization
        coordinates = np.zeros((self.batch_size, seq_len, 3))
        sequence = np.zeros((self.batch_size, str_len, self.labels[0].shape[1]))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample with padding
            seq_end = self.data[ID].shape[0]
            coordinates[i, :seq_end, :] = self.data[ID]

            # Get the sequence
            str_end = self.labels[ID].shape[0]
            sequence[i, :str_end, :] = self.labels[ID]

        return (coordinates[:, :-1,:], sequence), coordinates[:, 1:, :]

def extract_true_output(output):
    x = output[:,:,0]
    y = output[:,:,1]
    e = output[:,:,2]
    return x, y, e
  
def loss_function(output_mixtures):
    def loss(y_true, y_pred):
        # extract the data
        e, pi, mu1, mu2, std1, std2, rho = extract_pred_output(y_pred, output_mixtures)
        xs, ys, es = extract_true_output(y_true)
        
        xs = tf.expand_dims(xs, axis=2)
        ys = tf.expand_dims(ys, axis=2)

        # calculating the probability
        crho = 1 - tf.square(rho)
        xz = (xs - mu1) / std1
        yz = (ys - mu1) / std1
        Z = tf.square(xz) + tf.square(yz) - 2 * rho * xz * yz
        N = 1 / (2 * np.pi * std1 * std2 * tf.sqrt(crho)) * tf.exp(-Z / (2 * crho))

        prob_n = tf.reduce_sum(pi * N, axis=2)
        prob_e = e if es == 1 else 1 - e
        return tf.reduce_mean(-tf.log(prob_n + epsilon) - tf.log(prob_e))
    return loss

def main(_):
    # Datasets
    data, labels, translation = DataGenerator.load_dataset()

    # Generators
    training_generator = DataGenerator(range(len(data)), data, labels, batch_size=FLAGS.batch_size)

    # Training
    checkpoint_path = os.path.join(FLAGS.output_dir, "handwriting.ckpt")
    cp_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_path, save_weights_only=True,
        verbose=1)

    model = create_model(
        num_letters=len(translation), 
        window_mixtures=FLAGS.window_mixtures,
        output_mixtures=FLAGS.output_mixtures
    )
    model.compile(optimizer="adam",
        loss=loss_function(output_mixtures=FLAGS.output_mixtures)
    )
    model.fit_generator(
        generator=training_generator, 
        callbacks=[cp_callback],
        epochs=FLAGS.epochs
    )

if __name__ == "__main__":
    tf.app.run()