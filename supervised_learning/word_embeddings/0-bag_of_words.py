#!/usr/bin/env python3
"""Create bag-of-words embeddings."""

import re

import numpy as np


def bag_of_words(sentences, vocab=None):
    """Create a bag-of-words embedding matrix.

    Args:
        sentences: List of sentences to analyze.
        vocab: Optional list of vocabulary terms to include.

    Returns:
        A tuple containing the embedding matrix and its feature names.
    """
    tokenized = [re.findall(r"(?u)\b\w\w+\b", sentence.lower())
                 for sentence in sentences]
    features = list(vocab) if vocab is not None else sorted(
        {word for sentence in tokenized for word in sentence})
    feature_index = {word: index for index, word in enumerate(features)}
    embeddings = np.zeros((len(sentences), len(features)), dtype=int)
    for row, sentence in enumerate(tokenized):
        for word in sentence:
            if word in feature_index:
                embeddings[row, feature_index[word]] += 1
    return embeddings, features
