#!/usr/bin/env python3
"""Load and tokenize the TED Portuguese-to-English dataset."""

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """Prepare TED translation data and its sub-word tokenizers."""

    def __init__(self):
        """Load the training and validation splits and build tokenizers."""
        data = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            as_supervised=True,
        )
        self.data_train = data["train"]
        self.data_valid = data["validation"]
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """Build Portuguese and English sub-word tokenizers from ``data``."""
        tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy().decode("utf-8") for pt, _ in data),
            target_vocab_size=2 ** 15,
        )
        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy().decode("utf-8") for _, en in data),
            target_vocab_size=2 ** 15,
        )
        return tokenizer_pt, tokenizer_en
