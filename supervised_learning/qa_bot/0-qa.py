#!/usr/bin/env python3
"""Module for answering questions using a BERT QA model."""

import numpy as np
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """Find an answer to a question within a reference document.

    Args:
        question (str): The question to answer.
        reference (str): The reference document containing the answer.

    Returns:
        str: The extracted answer, or None if no answer is found.
    """
    tokenizer = BertTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad"
    )

    model = hub.load(
        "https://tfhub.dev/see--/bert-uncased-tf2-qa/1"
    )

    inputs = tokenizer.encode_plus(
        question,
        reference,
        add_special_tokens=True,
        return_tensors="tf",
        max_length=512,
        truncation=True
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    outputs = model(
        [
            input_ids,
            attention_mask,
            token_type_ids
        ]
    )

    start_scores = outputs[0][0]
    end_scores = outputs[1][0]

    start_index = int(np.argmax(start_scores))
    end_index = int(np.argmax(end_scores))

    if start_index == 0 or end_index == 0:
        return None

    if end_index < start_index:
        return None

    tokens = tokenizer.convert_ids_to_tokens(
        input_ids[0][start_index:end_index + 1]
    )

    answer = tokenizer.convert_tokens_to_string(tokens).strip()

    if not answer:
        return None

    return answer
