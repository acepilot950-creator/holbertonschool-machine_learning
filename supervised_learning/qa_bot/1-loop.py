#!/usr/bin/env python3
"""Module that creates a simple question and answer loop."""


def main():
    """Run the interactive question and answer loop."""
    exit_words = {"exit", "quit", "goodbye", "bye"}

    while True:
        question = input("Q: ")

        if question.lower() in exit_words:
            print("A: Goodbye")
            break

        print("A:")


if __name__ == "__main__":
    main()
