#!/usr/bin/env python3
"""Convert a gensim Word2Vec model to a Keras embedding layer."""


def gensim_to_keras(model):
    """Return a trainable Keras embedding initialized from ``model``.

    Args:
        model: A trained gensim Word2Vec model.

    Returns:
        A trainable Keras Embedding layer.
    """
    return model.wv.get_keras_embedding(True)
