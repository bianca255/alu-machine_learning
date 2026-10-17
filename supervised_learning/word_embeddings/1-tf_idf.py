#!/usr/bin/env python3
"""Create TF-IDF sentence embeddings."""

import re

import numpy as np


def tf_idf(sentences, vocab=None):
    """Create a TF-IDF embedding matrix.

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
    counts = np.zeros((len(sentences), len(features)), dtype=float)
    for row, sentence in enumerate(tokenized):
        for word in sentence:
            if word in feature_index:
                counts[row, feature_index[word]] += 1
    document_frequency = np.count_nonzero(counts, axis=0)
    idf = np.log((1 + len(sentences)) / (1 + document_frequency)) + 1
    embeddings = counts * idf
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = np.divide(embeddings, norms,
                           out=np.zeros_like(embeddings), where=norms != 0)
    return embeddings, features
