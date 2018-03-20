#!/usr/bin/env python3
import itertools
import pickle

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.python.framework.errors_impl import OutOfRangeError

import visualizations as viz

if False:
    from model import model_fn
    from tfinput import test_input_fn
else:
    from oxford_model import model_fn
    from oxtfinput import test_input_fn


def ensemble_predictions_conf(modeldirs, op=viz.by_mean, dataset='Cifar10',
                              title='24W Cifar10 Confusion Matrix'):
    # return a dictionary of ensemble accuracies at iter
    if dataset == 'Cifar10':
        params = {
            'img_dim': [32, 32, 3],
            'y_size': 10,
            'learning_rate': .0001
        }
    elif dataset == 'Oxford':
        params = {
            'img_dim': [500, 500, 3],
            'y_size': 17,
            'learning_rate': .001
        }

    models = []
    for i in range(len(modeldirs)):
        models.append(['worker' + str(i), tf.estimator.Estimator(
            model_fn=model_fn, params=params, model_dir=modeldirs[i]
        )])

    pred_cls, true_cls = viz.ensemble_accuracy(
        input_fn=test_input_fn, models=models,
        print_results=True, operation=op, output_cls=True)
    if dataset == 'Cifar10':
        with open('/home/euler/Cifar10/cifar-10-python/cifar-10-batches-py/batches.meta', 'rb') as fd:
            names = pickle.load(fd, encoding='bytes')
            names = [n.decode('utf-8') for n in names[b'label_names']]
    elif dataset == 'Oxford':
        names = ['Daffodil', 'Snow Drop', 'Lily Valley', 'Bluebell', 'Crocus', 'Iris', 'Tiger Lily',
                 'Tulip', 'Fritillary', 'Sunflower', 'Daisy', 'Colt\'s Foot', 'Dandelion', 'Cow\'s Lip',
                 'Buttercup', 'Windflower', 'Pansy']

    cm = confusion_matrix(true_cls, pred_cls)


    plot_confusion_matrix(cm,
                          classes=names,
                          normalize=False,
                          title=title)

    plt.show()

    return cm, names


def failures(input_fn, models=None, print_results=True, operation=None, batchnum=0, batchsize=32):
    # get true classes:
    sess = tf.Session()
    image_batch, label_batch = input_fn()
    labels = None
    i = 0
    while i < batchnum + 1:
        try:
            imgcls = sess.run(tf.reshape(image_batch['x'], [-1, 32, 32, 3])) # @hardcoded
            bincls = sess.run(label_batch)
        except OutOfRangeError:
            break
        img = imgcls
        labels = np.argmax(bincls, axis=1)
        i += 1
    my_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': imgcls},
        y=bincls,
        num_epochs=1,
        shuffle=False
    )

    collection = []
    accuracies = {}
    for model in models:
        name = model[0]
        model = model[1]
        assert (isinstance(model, tf.estimator.Estimator))
        prediction = model.predict(input_fn=my_input_fn)
        i = 0
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
        accuracies[op.__name__] = np.sum(val) / len(val)
        if print_results:
            print('metric: %s' % op.__name__)
            print('ensemble accuracy %s' % accuracies[op.__name__])

    return img, val


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Modified slightly from sklearn docs:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    if np.any(cm < 0):
        # rescale for better visualization
        absmax = np.max(np.abs(cm).flatten())
        plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=-absmax, vmax=absmax)
    else:
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def by_mean(softmaxes):
    # how to evaluate ensemble
    means = np.mean(softmaxes, axis=0)
    argmaxes = np.argmax(means, axis=1)

    return argmaxes


def open_cifar10(image):
    pass


def open_oxford(image):
    pass


if __name__ == '__main__':
    from glob import glob
    directory = 'oxmodels/ensemble_num*_iter10000'
    glb = glob(directory)
    csv = 'none.csv'
    fig = plt.figure(0)
    out, names = ensemble_predictions_conf(glb, title='9W Oxford Confusion Matrix', dataset='Oxford')
    fig.savefig('outputs/oxford_confusion_matrix.png')
    fig = plt.figure(1)
    out2, _ = ensemble_predictions_conf(glb, title='Oxford Best Individual', op='best_worker', dataset='Oxford')
    fig.savefig('outputs/oxford_confusion_matrix_bestworker.png')

    fig = plt.figure(2)
    plot_confusion_matrix(out - out2,
                          title='Confusion Difference',
                          classes=names,
                          cmap=plt.cm.bwr_r)
    plt.show()
    fig.savefig('outputs/oxford_conf_mat_difference.png')