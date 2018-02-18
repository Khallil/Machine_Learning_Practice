#-*- coding: utf-8 -*-

# Doudou Khallil
# Custom Deep Neural Net with Tensorflow tutorials

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import pandas as pd
import argparse


TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']

parser = argparse.ArgumentParser()
parser.add_argument('-b', default=100, type=int, help='batch size')
parser.add_argument('-e', default=1000, type=int,
                    help='number of training steps')

# download_data()
    #1 On récupère les DATA dans 2 csv, train et test
# load_data()
    #1 On load les data dans 4 list (train_x,train_y,test_x,test_y)
# get_iterator_from_data()
    #1 Converti le dataset en tf.Dataset
    #2 Retourne iterator de batches du tf.Dataset
# get_dataset_from_data()
    #1 Converti le dataset en tf.Dataset
    #2 Retourne un dataset batché
# my_model()
    #1 On crée l'INPUT LAYER en ajoutant features dans features_column 
    #2 On crée les HIDDEN LAYERS et définie leurs paramères
    #3 On crée l'OUTPUT LAYER en passant le nombres de classes
    #4 On définie le mode PREDICTION
    #5 On définie le mode EVALUATE
    #5.1   On définie la LOSS FUNCTION
    #5.2   On définie un METRIC
    #6 On définie le mode TRAIN
    #6.1   On définie l'OPTIMIZER
    #6.2   On définie la TRAINING OPERATION
# main ()
    # On définie la forme des FEATURES_COLUMNS
    # On définie la forme du RESEAU DE NEURONES, input_l,hidden_ls,output_l
    # On lance le TRAINING
    # On lance l' EVALUATE

def download_data():
    # keras télécharge et met le fichier dans ~/.keas/datasets/
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)
    
    return train_path, test_path

def load_data(y_name='Species'):
    train_path, test_path = download_data()

    # header = 0 : row 0 est la row des names, après on précise avec names=
    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)
    return (train_x, train_y), (test_x, test_y)

# On crée un dataset avec les features et label et batch_size
def get_iterator_from_data(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()

def get_dataset_from_data(features, labels, batch_size):
    features=dict(features)
    inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(batch_size)
    return dataset

# MY MODEL - Definition des LAYERS et des MODES
def my_model(features, labels, mode, params):
# Input Layer en mettant les features row dans la forme feature_columns
    net = tf.feature_column.input_layer(features, params['feature_columns'])

    # Build the hidden layers, sized according to the 'hidden_units' param.
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)   # activation fonction = Relu

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None) # aucun activation fonction

    # Defini le mode Predict
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits), # softmax
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # On définie la loss fonction (softmax) pour evaluate et training
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # On définie un metric accuracy pour plus d'info de l'eval
    accuracy = tf.metrics.accuracy(labels=labels,
                                predictions=predicted_classes,
                                name='acc_op')

    metrics = {'accuracy': accuracy} 
    tf.summary.scalar('accuracy', accuracy[1]) # pour TensorBoard
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN
    # On définie l'optimizer Adagrad avec alpha = 0.1
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    # On définie la training operation
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(argv):
    args = parser.parse_args(argv[1:])

    (train_x,train_y), (test_x, test_y) = load_data()
    # on créer les features colonnes en fonctions des noms des différents inputs
    my_feature_columns = [] # python.list
    # les keys sont ajouté à l'archiecture de la liste grâce a pd.read_csv
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params={
            'feature_columns': my_feature_columns,
            'hidden_units': [10, 10],
            'n_classes': 3,
        }
    )
    # Train the Model.
    classifier.train(
        input_fn=lambda:get_iterator_from_data(train_x, train_y, args.b),
        steps=args.e)
    
    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:get_dataset_from_data(test_x, test_y, args.b))
    
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

if __name__ == '__main__':
    tf.app.run(main)