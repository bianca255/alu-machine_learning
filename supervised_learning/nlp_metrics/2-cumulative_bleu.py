#!/usr/bin/env python3
"""Calculate cumulative n-gram BLEU scores."""

import math
from collections import Counter
from functools import reduce
from operator import mul


def _ngrams(sentence, n):
    """Return the n-grams in a tokenized sentence."""
    return [tuple(sentence[index:index + n])
            for index in range(len(sentence) - n + 1)]


def _brevity_penalty(references, sentence):
    """Return the BLEU brevity penalty for a candidate sentence."""
    if not sentence:
        return 0.0
    candidate_length = len(sentence)
    reference_length = min(
        (len(reference) for reference in references),
        key=lambda length: (abs(length - candidate_length), length))
    if candidate_length >= reference_length:
        return 1.0
    return math.exp(1 - reference_length / candidate_length)


def _precision(references, sentence, n):
    """Return clipped precision for one n-gram order."""
    candidate_ngrams = _ngrams(sentence, n)
    if not candidate_ngrams:
        return 0.0
    candidate_counts = Counter(candidate_ngrams)
    reference_counts = Counter()
    for reference in references:
        for ngram, count in Counter(_ngrams(reference, n)).items():
            reference_counts[ngram] = max(reference_counts[ngram], count)
    matches = sum(min(count, reference_counts[ngram])
                  for ngram, count in candidate_counts.items())
    return matches / len(candidate_ngrams)


def cumulative_bleu(references, sentence, n):
    """Calculate cumulative BLEU through the largest order ``n``."""
    if not references or not sentence or n < 1:
        return 0.0
    precisions = [_precision(references, sentence, order)
                  for order in range(1, n + 1)]
    if any(precision == 0 for precision in precisions):
        return 0.0
    product = reduce(mul, precisions, 1.0)
    geometric_mean = product ** (1 / n)
    return _brevity_penalty(references, sentence) * geometric_mean
