#!/usr/bin/env python3
"""Wrap translation encoding in TensorFlow and prepare encoded datasets."""

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """Prepare tokenized TED translation datasets."""

    def __init__(self):
        """Load data, create tokenizers, and encode both dataset splits."""
        data = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            as_supervised=True,
        )
        self.data_train = data["train"]
        self.data_valid = data["validation"]
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """Build sub-word tokenizers from the training data."""
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
        """Encode a translation pair with start and end tokens."""
        pt_tokens = [self.tokenizer_pt.vocab_size]
        pt_tokens.extend(self.tokenizer_pt.encode(pt.numpy().decode("utf-8")))
        pt_tokens.append(self.tokenizer_pt.vocab_size + 1)
        en_tokens = [self.tokenizer_en.vocab_size]
        en_tokens.extend(self.tokenizer_en.encode(en.numpy().decode("utf-8")))
        en_tokens.append(self.tokenizer_en.vocab_size + 1)
        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """TensorFlow wrapper around ``encode`` with dynamic output shapes."""
        pt_tokens, en_tokens = tf.py_function(
            self.encode,
            [pt, en],
            [tf.int64, tf.int64],
        )
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])
        return pt_tokens, en_tokens
