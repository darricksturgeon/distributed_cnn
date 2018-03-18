import argparse
import os
import re

import pandas as pd
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
from scipy.stats import mode
from tensorflow.python.framework.errors_impl import OutOfRangeError

from model import model_fn
from tfinput import test_input_fn


def cli_interface():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', '-c', help='name of csv')
    parser.add_argument('--dir', '-d', help='location of models')
    parser.add_argument('--print', action='store_true', help='print results as well')
    parser.add_argument('--plot', '-p', action='store_true', help='plot results')

    return parser.parse_args()


def main():
    args = cli_interface()

    directory = args.dir
    csv = args.csv
    plot = args.plot
    df = read_ensemble_series(directory)

    df.to_csv(csv)


def read_ensemble_series(directory):

    regex = re.compile(r'ensemble_num[0-9]*_iter([0-9]*)')
    matches = [regex.match(check) for check in os.listdir(directory)]

    idx = {}
    for item in matches:
        grp = int(item.groups()[0])
        idx[grp] = idx.get(grp, []) + [directory + '/' + item.string]

    acc = {}
    for key in sorted(idx.keys()):
        idx[key] = sorted(idx[key], key=numeric_sort)
        acc[key] = read_ensemble(idx[key])

    df = pd.DataFrame.from_dict(acc, orient='index')
    df['iterations'] = df.index
    return df


def read_ensemble(modeldirs):
    # return a dictionary of ensemble accuracies at iter

    cifar_params = {
        'img_dim': [32, 32, 3],
        'y_size': 10,
        'learning_rate': .001
    }

    models = []
    for i in range(len(modeldirs)):
        models.append(['worker' + str(i),  tf.estimator.Estimator(
            model_fn=model_fn, params=cifar_params, model_dir=modeldirs[i]
        )])

    accuracy = ensemble_accuracy(
        input_fn=test_input_fn, models=models,
        print_results=True, operation=(by_mean, by_mode, by_certainty))

    return accuracy


def ensemble_accuracy(input_fn, models=None, print_results=True, operation=None):

    # get true classes:
    sess = tf.Session()
    _, label_batch = input_fn()
    labels = None
    while True:
        try:
            bincls = sess.run(label_batch)
        except OutOfRangeError:
            break
        lab = np.argmax(bincls, axis=1)
        if labels is None:
            labels = lab
        else:
            labels = np.hstack((labels, lab))

    collection = []
    accuracies = {}
    for model in models:
        name = model[0]
        model = model[1]
        assert (isinstance(model, tf.estimator.Estimator))
        prediction = model.predict(input_fn=input_fn)
        softmax = [p for p in prediction]

        collection.append(softmax)
        mdl_pred = np.argmax(softmax, axis=1)
        val = np.where(labels == mdl_pred, 1, 0)
        accuracies[name] = np.sum(val) / len(val)
        if print_results:
            print(name + ' accuracy: %s' % accuracies[name])

    collection = np.array(collection)
    if not hasattr(operation, '__iter__'):
        operation = (operation,)

    for op in operation:
        results = op(collection)
        val = np.where(results == labels, 1, 0)
        accuracies[op.__name__] = np.sum(val)/len(val)
        if print_results:
            print('metric: %s' % op.__name__)
            print('ensemble accuracy %s' % accuracies[op.__name__])

    return accuracies


def by_mean(softmaxes):
    # how to evaluate ensemble
    means = np.mean(softmaxes, axis=0)
    argmaxes = np.argmax(means, axis=1)

    return argmaxes


def by_certainty(softmaxes):
    modes = np.max(softmaxes, axis=0)
    argmaxes = np.argmax(modes, axis=1)

    return argmaxes


def by_mode(softmaxes):
    preds = np.argmax(softmaxes, axis=2)
    modes = mode(preds, axis=0).mode.flatten()

    return modes


def numeric_sort(x):
    regex = re.compile('(\d+)')
    nums = regex.split(x)
    return [int(y) if y.isdigit() else y for y in nums]


def plot_helper(df):
    if isinstance(df, str):
        df = pd.read_csv(df, index_col=0)

    for item in df.columns:
        if item == 'iterations':
            continue
        if 'worker' in item:
            plt.plot(df['iterations'], np.exp(df[item]), '--')
        else:
            plt.plot(df['iterations'], np.exp(df[item]))
    plt.show()


if __name__ == '__main__':
    main()
    #plot_helper('test.csv')

    #cifar_params = {
    #    'img_dim': [32, 32, 3],
    #    'y_size': 10,
    #    'learning_rate': .001
    #}
    #models = []
    #for i in range(6):
    #    models.append(['task' + str(i),  tf.estimator.Estimator(
    #        model_fn=model_fn, params=cifar_params, model_dir='ensemble%s' % i
    #    )])
#
    #ensemble_accuracy(test_input_fn, models, operation=(by_mean, by_mode, by_certainty))
