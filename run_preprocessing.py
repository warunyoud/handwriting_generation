import os
import html
import xml.etree.ElementTree as ET

import numpy as np
import tensorflow as tf
import json

""" Output 
format: [[x_1, y_1, e_1], [x_2, y_2, e_2], ..., [x_n, y_n, e_n]]
    x: int. Offset from previous input in x.
    y: int. Offset from previous input in y.
    e: bool. Whether it is the end of a stroke.
"""


flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "output_dir", "data/training",
    "The output dir.")

## Optional parameters
flags.DEFINE_string(
    "data_dir", "data/original",
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

def get_char_mapping(charset):
    mapping = {c: i+1 for i, c in enumerate(sorted(charset))}
    mapping["<NULL>"] = 0
    return mapping

def get_textline_vectors(textline, char_mapping):
    return np.array([[char_mapping[c] == i for i in range(len(char_mapping))] for c in textline])

def clean_stroke(points, threshold=1000):
    if points.shape[0] < 3:
        return points

    offsets = np.zeros((points.shape[0], points.shape[1] - 1))
    offsets[1:, :] = points[1:,:-1] - points[:-1,:-1]
    offsets2 = np.zeros(offsets.shape)
    offsets2[1:, 0] = float("inf") 
    offsets2[2:, :] = points[2:,:-1] - points[-2,:-1]
    return points[np.logical_or(np.linalg.norm(offsets, axis=1) <= threshold,
        np.linalg.norm(offsets2, axis=1) <= threshold)]

def is_new_line(stroke, last_line, threshold=2000):
    last_x = last_line[-1][0]
    new_x = stroke[0, 0]
    return last_x - new_x > threshold

def extract_data_from_xml(file_path, charset):
    root = ET.parse(file_path)
    transcription = root.findall("Transcription")
    if not transcription:
        return [], []
    textlines = [html.unescape(item.get("text")) for item in transcription[0].findall("TextLine")]
    charset |= set("".join(textlines))

    strokes = [s.findall("Point") for s in root.findall("StrokeSet")[0].findall("Stroke")]
    strokes_by_line = []
    for stroke in strokes:
        points = np.array([[int(point.get("x")), int(point.get("y")), 0] for point in stroke])
        points = clean_stroke(points)
        points[-1, 2] = 1
        if len(strokes_by_line) == 0 or is_new_line(points, strokes_by_line[-1]):
            strokes_by_line.append(list(points))
        else:
            strokes_by_line[-1] += list(points)
    
    offsets = []
    for line in strokes_by_line:
        line = np.array(line)
        line[1:,:-1] =  line[1:,:-1] - line[:-1,:-1]
        line[0,:-1] = 0
        offsets.append(line)

    if len(offsets) == len(textlines):
        return offsets, textlines
    else:
        return [], []

def main():
    charset = set()
    data = []
    textlines = []
    for root, _, files in os.walk(FLAGS.data_dir):
        for f in files:
            fname, ext = os.path.splitext(f)
            if ext == ".xml":
                extract_data, extract_textlines = extract_data_from_xml(os.path.join(root, f), charset)
                data += extract_data
                textlines += extract_textlines
    print(charset)
    mapping = get_char_mapping(charset)
    labels = [get_textline_vectors(textline, mapping) for textline in textlines]
    
    np.save(os.path.join(FLAGS.output_dir, "data"), np.array(data))
    np.save(os.path.join(FLAGS.output_dir, "labels"), np.array(labels))
    with open(os.path.join(FLAGS.output_dir, "translation.json"), "w") as outfile:
        json.dump(mapping, outfile)

if __name__ == "__main__":
    main()