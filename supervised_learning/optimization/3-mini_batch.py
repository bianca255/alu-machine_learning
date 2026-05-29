#!/usr/bin/env python3
"""Mini-batch gradient descent training module."""
import numpy as np
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """Train a neural network using mini-batch gradient descent.

    Args:
        X_train: numpy.ndarray of shape (m, 784), training inputs.
        Y_train: numpy.ndarray of shape (m, 10), training labels (one-hot).
        X_valid: numpy.ndarray of shape (m, 784), validation inputs.
        Y_valid: numpy.ndarray of shape (m, 10), validation labels (one-hot).
        batch_size: number of data points per mini-batch.
        epochs: number of full passes through the training set.
        load_path: path to load the model checkpoint from.
        save_path: path to save the trained model checkpoint.

    Returns:
        The path where the model was saved.
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        m = X_train.shape[0]

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
                        print("\t\tAccuracy: {}".format(step_acc))

        return saver.save(sess, save_path)
