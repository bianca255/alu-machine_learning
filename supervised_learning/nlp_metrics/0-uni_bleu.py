#!/usr/bin/env python3
"""Calculate unigram BLEU scores without external NLP libraries."""

import math
from collections import Counter


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


def uni_bleu(references, sentence):
    """Calculate the unigram BLEU score for a candidate sentence."""
    if not references or not sentence:
        return 0.0
    candidate_counts = Counter(sentence)
    reference_counts = Counter()
    for reference in references:
        for word, count in Counter(reference).items():
            reference_counts[word] = max(reference_counts[word], count)
    matches = sum(min(count, reference_counts[word])
                  for word, count in candidate_counts.items())
    precision = matches / len(sentence)
    return _brevity_penalty(references, sentence) * precision
