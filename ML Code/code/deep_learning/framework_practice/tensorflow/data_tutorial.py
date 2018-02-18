#-*- coding: utf-8 -*-

# Doudou Khallil
# Data Tutorial

import numpy as np
import tensorflow as tf
import pandas as pd

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

# Format de donnée dans les csv
# metainformation at row 0# n_row | n_features | label0 | label1 | label2
# SepalL | SepalW | PetalL | PetalW | label_id

def maybe_download():
    # keras télécharge et met le fichier dans ~/.keas/datasets/
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)
    
    return train_path, test_path

"""Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
def load_data(y_name='Species'):
    train_path, test_path = maybe_download()

    # header = 0 : row 0 est la row des names, après on précise avec names=
    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)
    return (train_x, train_y), (test_x, test_y)


