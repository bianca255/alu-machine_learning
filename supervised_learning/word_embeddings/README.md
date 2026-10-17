# Word Embeddings

This directory contains sentence and word embedding exercises.

## Files

- `0-bag_of_words.py`: builds count-based bag-of-words embeddings.
- `1-tf_idf.py`: builds TF-IDF embeddings with L2 normalization.
- `2-word2vec.py`: trains a gensim Word2Vec model.
- `3-gensim_to_keras.py`: converts a trained gensim model to a trainable Keras embedding layer.
- `4-fasttext.py`: trains a gensim FastText model.
- `5-elmo`: records the statements that are trained when creating an ELMo embedding model.

The `bag_of_words` and `tf_idf` functions accept a list of sentences and an optional vocabulary. When `vocab` is `None`, features are collected from the input sentences in sorted order. When a vocabulary is supplied, its order is preserved and out-of-vocabulary words are ignored.

Word2Vec and FastText accept tokenized sentences and expose the assignment parameters for embedding size, context window, minimum count, negative sampling, training mode, epochs, random seed, and worker count.
