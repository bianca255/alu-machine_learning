#!/usr/bin/env python3
"""Train a gensim Word2Vec model."""

from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """Create and train a Word2Vec model.

    Args:
        sentences: Tokenized sentences used for training.
        size: Embedding dimensionality.
        min_count: Minimum word frequency.
        window: Maximum context window distance.
        negative: Number of negative samples.
        cbow: Use CBOW when true, or skip-gram when false.
        iterations: Number of training epochs.
        seed: Random seed.
        workers: Number of worker threads.

    Returns:
        The trained Word2Vec model.
    """
    arguments = {
        "vector_size": size,
        "min_count": min_count,
        "window": window,
        "negative": negative,
        "sg": 0 if cbow else 1,
        "epochs": iterations,
        "seed": seed,
        "workers": workers,
    }
    try:
        return Word2Vec(sentences, **arguments)
    except TypeError:
        arguments["size"] = arguments.pop("vector_size")
        arguments["iter"] = arguments.pop("epochs")
        return Word2Vec(sentences, **arguments)
