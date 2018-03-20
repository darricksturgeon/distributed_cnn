import argparse
import os
import re
from collections import OrderedDict

import pandas as pd
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
from scipy.stats import mode
from tensorflow.python.framework.errors_impl import OutOfRangeError

if False:
    from model import model_fn
    from tfinput import test_input_fn
else:
    from oxford_model import model_fn
    from oxtfinput import test_input_fn

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


def read_ensemble(modeldirs, params='cifar10'):
    # return a dictionary of ensemble accuracies at iter
    if params == 'cifar10':
        params = {
            'img_dim': [32, 32, 3],
            'y_size': 10,
            'learning_rate': .0001
        }
    elif params == 'oxford':
        params = {
            'img_dim': [500, 500, 3],
            'y_size': 17,
            'learning_rate': .0001
        }

    models = []
    for i in range(len(modeldirs)):
        models.append(['worker' + str(i),  tf.estimator.Estimator(
            model_fn=model_fn, params=params, model_dir=modeldirs[i]
        )])

    accuracy = ensemble_accuracy(
        input_fn=test_input_fn, models=models,
        print_results=True, operation=(by_mean, by_mode, by_certainty))

    return accuracy


def ensemble_accuracy(input_fn, models=None, print_results=True, operation=None, output_cls=False):

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
    best_of = None
    best = 0.0
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
        if operation == 'best_worker':
            if best < accuracies[name]:
                best = accuracies[name]
                best_of = mdl_pred
                best_of_name = name
    if operation == 'best_worker':

        return best_of, labels


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
    if output_cls:
        return results, labels
    else:
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


def plot_helper(df, ratio=1, name=None, yrange=None):
    if isinstance(df, str):
        df = pd.read_csv(df, index_col=0)
    colors = ['r', 'b', 'g']

    amnt = int(len(df['iterations'])*(1 - ratio))

    for item in df.columns:
        if item == 'iterations':
            continue
        if 'worker' in item:
            plt.plot(df['iterations'][amnt:], df[item][amnt:], 'k--')
        else:
            plt.plot(df['iterations'][amnt:], df[item][amnt:], colors.pop() + '-')
    ax = plt.gca()
    ax.minorticks_on()
    ax.tick_params(labelright=True)
    ax.yaxis.set_ticks_position('both')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    i = 1
    while by_label.pop('worker%s' % i, False):
        i += 1
    by_label['workers'] = by_label.pop('worker0')
    if name:
        plt.title(name)
    if yrange:
        axes = plt.gca()
        axes.set_ylim(yrange)
    plt.legend(list(by_label.values()), list(by_label.keys()))
    plt.xlabel('iterations')
    plt.ylabel('test accuracy')
    plt.show()


def best_worker(df, ratio=1, name=None, yrange=None):
    if isinstance(df, str):
        df = pd.read_csv(df, index_col=0)
    colors = ['r', 'b', 'g']

    workers = df.select(lambda col: col.startswith('worker'), axis=1)

    df['best_worker'] = workers.max(axis=1)

    amnt = int(len(df['iterations'])*(1 - ratio))

    for item in df.columns:
        if item == 'iterations':
            continue
        if item == 'best_worker':
            plt.plot(df['iterations'][amnt:], df[item][amnt:], 'k--')
        elif 'worker' in item:
            pass
        else:
            plt.plot(df['iterations'][amnt:], df[item][amnt:], colors.pop() + '-')
    ax = plt.gca()
    ax.minorticks_on()
    ax.tick_params(labelright=True)
    ax.yaxis.set_ticks_position('both')
    handles, labels = plt.gca().get_legend_handles_labels()

    by_label = OrderedDict(zip(labels, handles))
    i = 0
    while by_label.pop('worker%s' % i, False):
        i += 1
    if name:
        plt.title(name)
    if yrange:
        axes = plt.gca()
        axes.set_ylim(yrange)
    plt.legend(list(by_label.values()), list(by_label.keys()))
    plt.xlabel('iterations')
    plt.ylabel('test accuracy')
    plt.show()


def comparison(df_dict, ratio=1, measure='mean', name=None, yrange=None):
    for key in df_dict.keys():
        if isinstance(df_dict[key], str):
            df_dict[key] = pd.read_csv(df_dict[key], index_col=0)

        workers = df_dict[key].select(lambda col: col.startswith('worker'), axis=1)

        df_dict[key]['best individual'] = workers.max(axis=1)

        amnt = int(len(df_dict[key]['iterations']) * (1 - ratio))

        for item in df_dict[key].columns:
            if item == 'iterations':
                continue
            elif 'worker' in item:
                pass
            elif measure in item:
                tmpdf = df_dict[key].rename(columns={item: key})
                plt.plot(df_dict[key]['iterations'][amnt:], tmpdf[key][amnt:])
    ax = plt.gca()
    ax.minorticks_on()
    ax.tick_params(labelright=True)
    ax.yaxis.set_ticks_position('both')
    plt.legend()
    if name:
        plt.title(name)
    else:
        plt.title(measure)
    if yrange:
        axes = plt.gca()
        axes.set_ylim(yrange)
    plt.xlabel('iterations')
    plt.ylabel('test accuracy')
    plt.show()


if __name__ == '__main__':

    if False:
        main()
    if True:
        name='4 workers performance'
        fig = plt.figure(0)
        plot_helper('csvs/bigbatchmodel4.csv', 1, name=name, yrange=[.65, .825])
        fig.savefig('outputs/' + name + '.png')
    if True:
        name='4 worker ceiling'
        fig = plt.figure(1)
        best_worker('csvs/bigbatchmodel4.csv', 1, name=name, yrange=[.65, .825])
        fig.savefig('outputs/' + name + '.png')
    if False:
        df_dict = {
            '4 workers': 'csvs/bigbatchmodel4.csv',
            '8 workers': 'csvs/bigbatchmodel8.csv',
            '12 workers': 'csvs/bigbatchmodel12.csv',
            '16 workers': 'csvs/bigbatchmodel16.csv',
            '20 workers': 'csvs/bigbatchmodel20.csv',
            '24 workers': 'csvs/bigbatchmodel24.csv'
        }
        fig = plt.figure(2)
        comparison(df_dict, ratio=.75, measure='mean', name='ensemble mean', yrange=[.78, .82])
        fig.savefig('outputs/' + 'mean75pct.png')
        fig = plt.figure(3)
        comparison(df_dict, ratio=.75, measure='mode', name='ensemble mode', yrange=[.78, .82])
        fig.savefig('outputs/' + 'mode75pct.png')
        fig = plt.figure(4)
        comparison(df_dict, ratio=.75, measure='certainty', name='ensemble max', yrange=[.78, .82])
        fig.savefig('outputs/' + 'max75pct.png')
    if True:
        pass

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
