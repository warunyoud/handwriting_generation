import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def extract_pred_output(output, num_mixtures):
    step = num_mixtures
    e = output[:, :, 0]
    pi = output[:, :, 1:1+step]
    mu1 = output[:, :, 1+step:1+2*step]
    mu2 = output[:, :, 1+2*step:1+3*step]
    std1 = output[:, :, 1+3*step:1+4*step]
    std2 = output[:, :, 1+4*step:1+5*step]
    rho = output[:, :, 1+5*step:1+6*step]
    return e, pi, mu1, mu2, std1, std2, rho

def get_textline_vectors(textline, translation):
    return np.array([[translation[c] == i for i in range(len(translation))] for c in textline])

def get_sentence_from_vector(vector, translation):
    inv_translation = {}
    for key, value in translation.items():
        inv_translation[value] = key
    return [inv_translation[np.argmax(c)] for c in vector]

def plot_handwriting(points, sequence):
    coordinates = np.cumsum(points[:, :2], axis=0)
    start = 0
    fig, ax = plt.subplots()
    for i in range(1, len(points)):
        if points[i, 2] == 1:
            plt.plot(coordinates[start:i, 0], coordinates[start:i, 1], "b-", linewidth=2.0)
            start = i + 1
    # plt.plot(coordinates[:, 0], -coordinates[:, 1])
    plt.title(sequence)
    plt.gca().invert_yaxis()
    plt.show()
