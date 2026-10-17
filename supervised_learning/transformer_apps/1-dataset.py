#!/usr/bin/env python3
"""Encode TED translation examples as Transformer token sequences."""

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """Prepare TED data and encode Portuguese-English sentence pairs."""

    def __init__(self):
        """Load the dataset and create tokenizers from its training split."""
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
        """Build sub-word tokenizers from the supplied translation data."""
        tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy().decode("utf-8") for pt, _ in data),
            target_vocab_size=2 ** 15,
        )
        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy().decode("utf-8") for _, en in data),
            target_vocab_size=2 ** 15,
        )
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Encode a Portuguese-English pair with start and end tokens."""
        pt_tokens = [self.tokenizer_pt.vocab_size]
        pt_tokens.extend(self.tokenizer_pt.encode(pt.numpy().decode("utf-8")))
        pt_tokens.append(self.tokenizer_pt.vocab_size + 1)
        en_tokens = [self.tokenizer_en.vocab_size]
        en_tokens.extend(self.tokenizer_en.encode(en.numpy().decode("utf-8")))
        en_tokens.append(self.tokenizer_en.vocab_size + 1)
        return pt_tokens, en_tokens
