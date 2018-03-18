import tensorflow as tf


def model_fn(features, labels, mode, params):
    # enter training architecture, default is to train on data but can be overridden for predictions...
    # x = tf.placeholder_with_default(features, shape=[-1, np.prod(params.img_dim)], name='features')
    # y = tf.placeholder_with_default(labels, shape=[-1, np.prod(params.y_size)], name='labels')
    x = features['x']
    x_image = tf.reshape(x, [-1, *params['img_dim']])
    tf.Print(x_image, data=[x_image])

    training = (mode == tf.estimator.ModeKeys.TRAIN)

    graph = x_image

    graph = tf.layers.conv2d(graph, name='layer_conv2d_1', padding='same',
                          filters=16, kernel_size=3, activation=tf.nn.relu)
    graph = tf.layers.dropout(graph, rate=0.4, training=training)
    graph = tf.layers.max_pooling2d(graph, pool_size=2, name='layer_maxpool_p1_2', strides=2)
    graph = tf.layers.conv2d(graph, name='layer_conv2d_2', padding='same',
                              filters=16, kernel_size=3, activation=tf.nn.relu)
    graph = tf.layers.dropout(graph, rate=0.4, training=training)

    graph = tf.layers.max_pooling2d(graph, pool_size=2, name='layer_maxpool_p1_3', strides=2)
    graph = tf.layers.conv2d(graph, name='layer_conv2d_3', padding='same',
                             filters=16, kernel_size=3, activation=tf.nn.relu)
    graph = tf.layers.dropout(graph, rate=0.3, training=training)

    graph = tf.layers.max_pooling2d(graph, pool_size=2, name='layer_maxpool_p1_4', strides=2)
    graph = tf.layers.conv2d(graph, name='layer_conv2d_4', padding='same',
                             filters=16, kernel_size=5, activation=tf.nn.relu)
    graph = tf.layers.dropout(graph, rate=0.3, training=training)

    graph = tf.layers.max_pooling2d(graph, pool_size=2, name='layer_maxpool_p1_5', strides=2)
    graph = tf.layers.conv2d(graph, name='layer_conv2d_5', padding='same',
                             filters=16, kernel_size=5, activation=tf.nn.relu)
    graph = tf.layers.dropout(graph, rate=0.3, training=training)

    graph = tf.layers.max_pooling2d(graph, pool_size=2, name='layer_maxpool_p1_6', strides=2)
    graph = tf.layers.conv2d(graph, name='layer_conv2d_6', padding='same',
                             filters=16, kernel_size=5, activation=tf.nn.relu)
    graph = tf.layers.dropout(graph, rate=0.3, training=training)

    # graph = tf.layers.max_pooling2d(graph, pool_size=2, name='layer_maxpool_p1_2', strides=2)

    graph = tf.layers.flatten(graph, name='flatten')

    graph = tf.layers.dense(graph, name='fully_conn', units=256, activation=tf.nn.relu)
    graph = tf.layers.dropout(graph, rate=0.3, training=training)

    graph = tf.layers.dense(graph, name='fully_conn2', units=64, activation=tf.nn.relu)
    graph = tf.layers.dropout(graph, rate=0.3, training=training)

    graph = tf.layers.dense(graph, name='logits', units=params['y_size'], activation=None)

    logits = graph

    y_pred = tf.nn.softmax(logits=logits)

    # for predictions
    y_pred_cls = tf.argmax(y_pred, axis=1, name='prediction')

    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode, predictions=y_pred_cls)
    else:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        class_labels = tf.argmax(labels, axis=1)  # BinaryLabels to numeric
        loss = tf.reduce_mean(cross_entropy, name='myloss')
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        training = optimizer.minimize(loss, global_step=tf.train.get_global_step(), name='mygraph')
        metrics = {'accuracy': tf.metrics.accuracy(class_labels, y_pred_cls)}

        spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=training, eval_metric_ops=metrics)

    return spec
