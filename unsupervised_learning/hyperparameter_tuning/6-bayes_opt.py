#!/usr/bin/env python3
"""Optimize a machine learning model using GPyOpt."""
import GPyOpt
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
import numpy as np


def build_model(learning_rate, units, dropout_rate, l2_weight, batch_size):
    """Build and train the model."""
    (x_train, y_train), (x_val, y_val) = mnist.load_data()
    x_train, x_val = x_train / 255.0, x_val / 255.0

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(units, activation='relu',
                           kernel_regularizer=keras.regularizers.l2(l2_weight)),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(10, activation='softmax')
    ])

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Specify checkpoint file name including hyperparams
    checkpoint_name = "checkpoint_lr{:.4f}_u{}_dr{:.2f}_l2{:.4f}_bs{}.h5".format(
        learning_rate, int(units), dropout_rate, l2_weight, int(batch_size))
    
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
        keras.callbacks.ModelCheckpoint(checkpoint_name, save_best_only=True)
    ]

    history = model.fit(x_train, y_train, epochs=10, batch_size=int(batch_size),
                        validation_data=(x_val, y_val), callbacks=callbacks, verbose=0)
    
    # Return best validation loss (since GPyOpt minimizes)
    return min(history.history['val_loss'])

def objective_function(x):
    """Objective function for GPyOpt."""
    learning_rate = float(x[:, 0])
    units = int(x[:, 1])
    dropout_rate = float(x[:, 2])
    l2_weight = float(x[:, 3])
    batch_size = int(x[:, 4])
    
    loss = build_model(learning_rate, units, dropout_rate, l2_weight, batch_size)
    return np.array([[loss]])


if __name__ == '__main__':
    bounds = [
        {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-4, 1e-2)},
        {'name': 'units', 'type': 'discrete', 'domain': (32, 64, 128, 256)},
        {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.1, 0.5)},
        {'name': 'l2_weight', 'type': 'continuous', 'domain': (1e-5, 1e-2)},
        {'name': 'batch_size', 'type': 'discrete', 'domain': (32, 64, 128)}
    ]

    optimizer = GPyOpt.methods.BayesianOptimization(
        f=objective_function,
        domain=bounds,
        model_type='GP',
        acquisition_type='EI',
        maximize=False
    )

    optimizer.run_optimization(max_iter=30)
    
    # Save the report
    with open('bayes_opt.txt', 'w') as f:
        f.write("Best value: {}\n".format(optimizer.fx_opt))
        f.write("Best hyperparameters:\n")
        f.write(str(optimizer.x_opt))

    # Plot convergence
    optimizer.plot_convergence()
