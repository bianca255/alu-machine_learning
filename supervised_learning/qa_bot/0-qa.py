#!/usr/bin/env python3
"""Answer questions from a reference document with a TensorFlow Hub model."""

import tensorflow_hub as hub
from transformers import BertTokenizer


MODEL_URL = "https://tfhub.dev/see--/bert-uncased-tf2-qa/1"
TOKENIZER_NAME = "bert-large-uncased-whole-word-masking-finetuned-squad"


def question_answer(question, reference):
    """Return the answer span found in ``reference`` or ``None``."""
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)
    model = hub.load(MODEL_URL)
    encoded = tokenizer(question, reference, return_tensors="tf")
    result = model(encoded)

    start = int(result["start_logits"].numpy().argmax())
    end = int(result["end_logits"].numpy().argmax())
    if start > end:
        return None

    answer = tokenizer.decode(encoded["input_ids"][0][start:end + 1])
    if not answer or answer in {"[CLS]", "[SEP]"}:
        return None
    return answer
