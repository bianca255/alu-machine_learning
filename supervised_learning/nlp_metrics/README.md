# NLP Evaluation Metrics

This directory implements BLEU evaluation metrics without using NLTK.

- `0-uni_bleu.py` calculates clipped unigram precision with the BLEU brevity penalty.
- `1-ngram_bleu.py` calculates clipped precision for a requested n-gram order.
- `2-cumulative_bleu.py` calculates the geometric mean of clipped precisions from order 1 through `n`, then applies the brevity penalty.

Reference length uses the reference whose length is closest to the candidate, preferring the shorter reference on ties. Candidate n-grams are clipped by the maximum count observed in any reference.
