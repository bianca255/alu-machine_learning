#!/usr/bin/env python3
"""Full optimization model module."""
import numpy as np
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def model(Data_train, Data_valid, layers, activations,
          alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
          decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """Build, train, and save a neural network with all optimizations.

    Uses Adam optimization, mini-batch gradient descent, learning rate decay,
    and batch normalization.

    Args:
        Data_train: tuple (X_train, Y_train).
        Data_valid: tuple (X_valid, Y_valid).
        layers: list of node counts per layer.
        activations: list of activation functions per layer.
        alpha: learning rate.
        beta1: Adam first moment weight.
        beta2: Adam second moment weight.
        epsilon: small value to avoid division by zero.
        decay_rate: inverse time decay rate (decay_step=1).
        batch_size: mini-batch size.
        epochs: number of training epochs.
        save_path: path to save the model checkpoint.

    Returns:
        Path where the model was saved.
    """
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid
    nx = X_train.shape[1]
    classes = Y_train.shape[1]

    # ---- Build graph ----
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    # Hidden + output layers with batch norm on all but last
    prev = x
    for i, (n, act) in enumerate(zip(layers, activations)):
        if i < len(layers) - 1:
            prev = _create_batch_norm_layer(prev, n, act)
        else:
            init = tf.contrib.layers.variance_scaling_initializer(
                mode="FAN_AVG")
            prev = tf.layers.Dense(n, kernel_initializer=init)(prev)
            if act is not None:
                prev = act(prev)

    y_pred = prev
    tf.add_to_collection('y_pred', y_pred)

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_pred))
    tf.add_to_collection('loss', loss)

    correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.add_to_collection('accuracy', accuracy)

    global_step = tf.Variable(0, trainable=False)
    decayed_alpha = tf.train.inverse_time_decay(
        alpha, global_step, 1, decay_rate, staircase=True)
    train_op = tf.train.AdamOptimizer(
        decayed_alpha, beta1, beta2, epsilon).minimize(
            loss, global_step=global_step)
    tf.add_to_collection('train_op', train_op)

    m = X_train.shape[0]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        for epoch in range(epochs + 1):
            train_cost, train_acc = sess.run(
                [loss, accuracy], feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_acc = sess.run(
                [loss, accuracy], feed_dict={x: X_valid, y: Y_valid})

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_acc))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_acc))

            if epoch < epochs:
                X_shuf, Y_shuf = shuffle_data(X_train, Y_train)

                for step in range(0, m, batch_size):
                    X_batch = X_shuf[step:step + batch_size]
                    Y_batch = Y_shuf[step:step + batch_size]
                    sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                    step_num = step // batch_size + 1
                    if step_num % 100 == 0:
                        step_cost, step_acc = sess.run(
                            [loss, accuracy],
                            feed_dict={x: X_batch, y: Y_batch})
                        print("\tStep {}:".format(step_num))
                        print("\t\tCost: {}".format(step_cost))
                        print("\t\tAccuracy {}".format(step_acc))

        return saver.save(sess, save_path)


def _create_batch_norm_layer(prev, n, activation):
    """Helper: dense layer with batch normalization and activation.

    Args:
        prev: input tensor from the previous layer.
        n: number of nodes.
        activation: activation function to apply after batch norm.

    Returns:
        Activated output tensor.
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    Z = tf.layers.Dense(n, kernel_initializer=init)(prev)
    mean, variance = tf.nn.moments(Z, axes=[0])
    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)
    Z_norm = tf.nn.batch_normalization(Z, mean, variance, beta, gamma, 1e-8)
    if activation is None:
        return Z_norm
    return activation(Z_norm)
