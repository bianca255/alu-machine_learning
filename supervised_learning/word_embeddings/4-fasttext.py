#!/usr/bin/env python3
"""Train a gensim FastText model."""

from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5, negative=5, window=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """Create and train a FastText model.

    Args:
        sentences: Tokenized sentences used for training.
        size: Embedding dimensionality.
        min_count: Minimum word frequency.
        negative: Number of negative samples.
        window: Maximum context window distance.
        cbow: Use CBOW when true, or skip-gram when false.
        iterations: Number of training epochs.
        seed: Random seed.
        workers: Number of worker threads.

    Returns:
        The trained FastText model.
    """
    arguments = {
        "vector_size": size,
        "min_count": min_count,
        "negative": negative,
        "window": window,
        "sg": 0 if cbow else 1,
        "epochs": iterations,
        "seed": seed,
        "workers": workers,
    }
    try:
        return FastText(sentences, **arguments)
    except TypeError:
        arguments["size"] = arguments.pop("vector_size")
        arguments["iter"] = arguments.pop("epochs")
        return FastText(sentences, **arguments)
