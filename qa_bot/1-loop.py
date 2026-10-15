#!/usr/bin/env python3
"""Run an interactive question prompt."""


def main():
    """Read questions until the user enters a quit command."""
    while True:
        question = input("Q: ")
        if question.lower() in {"exit", "quit", "goodbye", "bye"}:
            print("A: Goodbye")
            return
        print("A:")


if __name__ == "__main__":
    main()
