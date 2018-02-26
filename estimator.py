import tensorflow as tf

import tfinput


def model_fn(features, labels, mode, params):
    # enter training architecture, default is to train on data but can be overridden for predictions...
    # x = tf.placeholder_with_default(features, shape=[-1, np.prod(params.img_dim)], name='features')
    # y = tf.placeholder_with_default(labels, shape=[-1, np.prod(params.y_size)], name='labels')
    x = features['x']
    x_image = tf.reshape(x, [-1, *params['img_dim']])

    training = (mode == tf.estimator.ModeKeys.TRAIN)

    graph = x_image

    # split 1
    graph1 = tf.layers.conv2d(graph, name='layer_conv2d_1', padding='same',
                              filters=8, kernel_size=5, activation=tf.nn.relu)
    graph1 = tf.layers.dropout(graph1, rate=0.3, training=training)

    graph1 = tf.layers.max_pooling2d(graph1, pool_size=2, name='layer_maxpool_p1_1', strides=2)
    graph1 = tf.layers.conv2d(graph1, name='layer_conv2d_2', padding='same',
                              filters=8, kernel_size=5, activation=tf.nn.relu)
    graph1 = tf.layers.dropout(graph1, rate=0.3, training=training)

    graph1 = tf.layers.max_pooling2d(graph1, pool_size=2, name='layer_maxpool_p1_2', strides=2)

    # split 2
    graph2 = tf.layers.max_pooling2d(graph, pool_size=2, name='layer_maxpool_1', strides=2)
    graph2 = tf.layers.conv2d(graph2, name='layer_conv2d_p2_1', padding='same',
                              filters=8, kernel_size=5, activation=tf.nn.relu)
    graph2 = tf.layers.dropout(graph2, rate=0.3)
    graph2 = tf.layers.max_pooling2d(graph2, pool_size=2, name='layer_maxpool_2', strides=2)
    graph2 = tf.layers.conv2d(graph2, name='layer_conv2d_p2_2', padding='same',
                              filters=8, kernel_size=5, activation=tf.nn.relu)
    graph2 = tf.layers.dropout(graph2, rate=0.3, training=training)

    graph = tf.concat([graph1, graph2], axis=-1, name='concat')
    graph = tf.layers.flatten(graph, name='flatten')

    graph = tf.layers.dense(graph, name='fully_conn', units=256, activation=tf.nn.relu)
    graph = tf.layers.dropout(graph, rate=0.3, training=training)

    graph = tf.layers.dense(graph, name='fully_conn2', units=64, activation=tf.nn.relu)
    graph = tf.layers.dropout(graph, rate=0.3, training=training)

    graph = tf.layers.dense(graph, name='logits', units=params['y_size'], activation=None)

    logits = graph

    y_pred = tf.nn.softmax(logits=logits)

    # for predictions
    y_pred_cls = tf.argmax(y_pred, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode, predictions=y_pred_cls)
    else:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        class_labels = tf.argmax(labels, axis=1)  # BinaryLabels to numeric
        loss = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        training = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        metrics = {'accuracy': tf.metrics.accuracy(class_labels, y_pred_cls)}

        spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=training, eval_metric_ops=metrics)

    return spec


def main():

    oxford_params = {
        'img_dim': [500, 500, 3],
        'y_size': 17,
        'learning_rate': .001
    }

    model = tf.estimator.Estimator(model_fn=model_fn, params=oxford_params,
                                   model_dir='./oxford_checkpoints3/')

    # enter estimator training
    model.train(input_fn=tfinput.train_input_fn, steps=1000)

    # evaluate
    results = model.evaluate(input_fn=tfinput.test_input_fn)

    print(results)


if __name__ == '__main__':
    main()
