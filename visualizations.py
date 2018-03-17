import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
from scipy.stats import mode
from tensorflow.python.framework.errors_impl import OutOfRangeError


def ensemble_accuracy(input_fn, models=None, print_results=True, operation=None):

    ens_acc = []
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
    for model in models:
        name = model[0]
        model = model[1]
        assert (isinstance(model, tf.estimator.Estimator))
        prediction = model.predict(input_fn=input_fn)
        softmax = [p for p in prediction]

        collection.append(softmax)
        if print_results:
            mdl_pred = np.argmax(softmax, axis=1)
            val = np.where(labels == mdl_pred, 1, 0)
            print(name + ' accuracy: %s' % (np.sum(val)/len(val)))

    collection = np.array(collection)
    if not hasattr(operation, '__iter__'):
        operation = (operation,)

    for op in operation:
        results = op(collection)
        if print_results:
            val = np.where(results == labels, 1, 0)
            print('metric: %s' % op.__name__)
            print('ensemble accuracy %s' % (np.sum(val)/len(val)))


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


if __name__ == '__main__':
    from model import model_fn
    from tfinput import test_input_fn

    cifar_params = {
        'img_dim': [32, 32, 3],
        'y_size': 10,
        'learning_rate': .001
    }
    models = []
    for i in range(6):
        models.append(['task' + str(i),  tf.estimator.Estimator(
            model_fn=model_fn, params=cifar_params, model_dir='ensemble%s' % i
        )])

    ensemble_accuracy(test_input_fn, models, operation=(by_mean, by_mode, by_certainty))
