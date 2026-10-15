#!/usr/bin/env python3
"""Answer questions by searching a corpus of reference documents."""

semantic_search = __import__("3-semantic_search").semantic_search
answer_question = __import__("0-qa").question_answer


def question_answer(corpus_path):
    """Answer questions using the most relevant document in ``corpus_path``."""
    while True:
        question = input("Q: ")
        if question.lower() in {"exit", "quit", "goodbye", "bye"}:
            print("A: Goodbye")
            return

        reference = semantic_search(corpus_path, question)
        answer = answer_question(question, reference) if reference else None
        if answer is None:
            answer = "Sorry, I do not understand the question."
        print("A: {}".format(answer))
