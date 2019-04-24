import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from modelling import create_model
from helpers import get_textline_vectors, extract_pred_output, plot_handwriting


epsilon = 1e-8

flags = tf.flags
FLAGS = flags.FLAGS

## Parameters
flags.DEFINE_string(
    "data_dir", "data/training",
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "output_dir", "output",
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_integer(
    "num_mixtures", 10,
    "The number of mixtures ... ")

def predict_next(e, pi, mu1, mu2, std1, std2, rho):
    coordinates = np.zeros(2)
    std12 = std1[0][-1] * std2[0][-1] * rho[0][-1]
    mixture = np.random.choice(np.arange(pi.shape[2]), p=pi[0][0])
    mean = [mu1[0][-1][mixture], mu2[0][-1][mixture]]
    cov = [[std1[0][-1][mixture]**2, std12[mixture]],
            [std12[mixture], std2[0][-1][mixture]**2]]
    coordinates = np.random.multivariate_normal(mean, cov, 1)[0]
    return coordinates[0], coordinates[1], np.random.binomial(1, e[0][-1])

def main(_):
    with open(os.path.join(FLAGS.data_dir, "translation.json")) as json_file:
        translation = json.load(json_file)

    latest = tf.train.latest_checkpoint(FLAGS.output_dir)

    # Generators
    model = create_model(        
        num_letters=len(translation), 
        num_mixtures=FLAGS.num_mixtures
    )
    model.load_weights(latest)

    my_str = "to stop Mr. Gaitskell from"
    sequence = get_textline_vectors(my_str, translation)
    points = [[0, 0, 0]]
    handwritings = []
    for i in range(50):
        output = model.predict(([points], [sequence]))
        e, pi, mu1, mu2, std1, std2, rho = extract_pred_output(output, FLAGS.num_mixtures)
        point = predict_next(e, pi, mu1, mu2, std1, std2, rho)
        points.append(point)
    plot_handwriting(np.array(points), my_str)
if __name__ == "__main__":
    # flags.mark_flag_as_required("output_dir")
    tf.app.run()