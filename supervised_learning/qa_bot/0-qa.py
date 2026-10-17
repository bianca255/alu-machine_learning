#!/usr/bin/env python3
"""Answer questions from a reference document with a TensorFlow Hub model."""

import tensorflow_hub as hub
import tensorflow as tf
from transformers import BertTokenizer


MODEL_URL = "https://tfhub.dev/see--/bert-uncased-tf2-qa/1"
TOKENIZER_NAME = "bert-large-uncased-whole-word-masking-finetuned-squad"


def question_answer(question, reference):
    """Return the answer span found in ``reference`` or ``None``."""
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)
    model = hub.load(MODEL_URL)

    question_tokens = tokenizer.tokenize(question)
    reference_tokens = tokenizer.tokenize(reference)
    tokens = (['[CLS]'] + question_tokens + ['[SEP]'] +
              reference_tokens + ['[SEP]'])
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    input_type_ids = ([0] * (len(question_tokens) + 2) +
                      [1] * (len(reference_tokens) + 1))
    inputs = {
        "input_ids": tf.constant([input_ids]),
        "input_mask": tf.constant([input_mask]),
        "input_type_ids": tf.constant([input_type_ids]),
    }
    result = model(inputs)

    if isinstance(result, dict):
        start_logits = result["start_logits"]
        end_logits = result["end_logits"]
    else:
        start_logits, end_logits = result[:2]
    start = int(tf.argmax(start_logits[0]).numpy())
    end = int(tf.argmax(end_logits[0]).numpy())
    if start > end:
        return None

    answer_tokens = tokens[start:end + 1]
    if any(token in {"[CLS]", "[SEP]"} for token in answer_tokens):
        return None
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    return answer if answer else None
