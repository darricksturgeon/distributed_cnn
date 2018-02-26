from collections import namedtuple

import tensorflow as tf
import numpy as np

import tfinput


def main():

    # graph layout plans:
    # input layer
    # maxpool maxpool maxpool layers
    # conv2d conv2d conv2d *x layers...
    # flatten layer
    # output layer

    # begin constructing network architecture
    img_dim = (500, 500, 3)  # will randomly crop inputs then have 3 color channels
    y_size = 17
    learning_rate = 1e-4

    # get sample of data to train on...
    x_batch, y_batch = tfinput.train_input_fn()

    # to modularize later, params will be static for given run, but dynamic in terms of testing.
    parameters = namedtuple('params', ['img_dim', 'y_size', 'learning_rate'])
    params = parameters(img_dim=img_dim, y_size=y_size, learning_rate=learning_rate)

    # download and format input data into binary tf objects
    tfinput.create_tfrecords()

    # enter training architecture, default is to train on data but can be overridden for predictions...
    x = tf.placeholder_with_default(x_batch['x'], shape=[None, np.prod(params.img_dim)], name='x')
    y = tf.placeholder_with_default(y_batch, shape=[None, np.prod(y_size)], name='y')

    x_image = tf.reshape(x, [-1, *params.img_dim])
    y_cls = tf.argmax(y, dimension=1)

    graph = x_image

    graph1 = tf.layers.conv2d(graph, name='layer_conv2d_1', padding='same',
                             filters=8, kernel_size=5, activation=tf.nn.relu)

    graph1 = tf.layers.max_pooling2d(graph1, pool_size=2, name='layer_maxpool_p1_1', strides=2)

    graph1 = tf.layers.conv2d(graph1, name='layer_conv2d_2', padding='same',
                              filters=16, kernel_size=5, activation=tf.nn.relu)

    graph1 = tf.layers.max_pooling2d(graph1, pool_size=2, name='layer_maxpool_p1_2', strides=2)

    #layer1 = graph

    graph2 = tf.layers.max_pooling2d(graph, pool_size=2, name='layer_maxpool_1', strides=2)

    #pool1 = graph

    graph2 = tf.layers.conv2d(graph2, name='layer_conv2d_p2_1', padding='same',
                             filters=32, kernel_size=5, activation=tf.nn.relu)

    graph2 = tf.layers.max_pooling2d(graph, pool_size=2, name='layer_maxpool_2', strides=2)

    graph2 = tf.layers.conv2d(graph2, name='layer_conv2d_p2_2', padding='same',
                              filters=16, kernel_size=5, activation=tf.nn.relu)

    #layer2 = graph

    #graph = tf.layers.max_pooling2d(graph, pool_size=2, name='layer_maxpool_2', strides=2)

    #pool2 = graph

    graph1 = tf.layers.flatten(graph1, name='flatten_1')

    graph2 = tf.layers.flatten(graph2, name='flatten_2')

    graph = tf.concat([graph1, graph2], axis=1)


    #flattened = graph

    graph = tf.layers.dense(graph, name='fully_conn', units=128, activation=tf.nn.relu)

    graph = tf.layers.dense(graph, name='logits', units=params.y_size, activation=None)

    logits = graph

    y_pred = tf.nn.softmax(logits=logits)

    # for predictions
    y_pred_cls = tf.argmax(y_pred, dimension=1)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)

    loss = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate).minimize(loss)


    # needs current scope...
    def print_test_accuracy():
        # Number of images in the test-set.
        num_test = 272  # @hardcoded this is the size of the testing set

        # Allocate an array for the predicted classes which
        # will be calculated in batches and filled into this array.
        images, labels = tfinput.test_input_fn()
        cls_pred = np.zeros(shape=num_test, dtype=np.int)

        i = 0
        while i < num_test:
            # The ending index for the next batch is denoted j.
            j = min(i + 64, num_test)

            # Get the images from the test-set between index i and j.

            # Create a feed-dict with these images and labels.
            data_nodes = {x: images['x'].eval(session=session),
                          y: labels.eval(session=session)}

            # Calculate the predicted class using TensorFlow.
            cls_pred[i:j] = session.run(y_pred_cls, feed_dict=data_nodes)

            # Set the start-index for the next batch to the
            # end-index of the current batch.
            if i == 0:
                test_labels = labels
            else:
                labels = tf.concat((test_labels, labels), axis=0)
            i = j

        # Convenience variable for the true class-numbers of the test-set.

        # Create a boolean array whether each image is correctly classified.
        y_true = np.argmax(test_labels.eval(session=session), axis=1)
        correct = (y_true == cls_pred)

        # Calculate the number of correctly classified images.
        # When summing a boolean array, False means 0 and True means 1.
        correct_sum = correct.sum()

        # Classification accuracy is the number of correctly classified
        # images divided by the total number of images in the test-set.
        acc = float(correct_sum) / num_test

        # Print the accuracy.
        msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
        print(msg.format(acc, correct_sum, num_test))


    # enter modeling
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        # run model for some epochs
        for i in range(5001):

            session.run(optimizer)
            if not (i % 100):
                print_test_accuracy()


if __name__ == '__main__':
    main()
