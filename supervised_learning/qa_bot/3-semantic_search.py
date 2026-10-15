#!/usr/bin/env python3
"""Find the document in a corpus that best matches a sentence."""

import glob
import os

from sentence_transformers import SentenceTransformer, util


def semantic_search(corpus_path, sentence):
    """Return the text of the corpus document most similar to ``sentence``."""
    paths = sorted(glob.glob(os.path.join(corpus_path, "*.md")))
    if not paths:
        return None

    documents = [open(path, encoding="utf-8").read() for path in paths]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode(sentence, convert_to_tensor=True)
    document_embeddings = model.encode(documents, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, document_embeddings)[0]
    best_index = int(similarities.argmax())
    return documents[best_index]
