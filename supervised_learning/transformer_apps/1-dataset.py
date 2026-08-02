#!/usr/bin/env python3
"""Dataset preparation module for Portuguese-to-English translation."""

import transformers
from setup import load_pt2en


class Dataset:
    """Load and prepare a Portuguese-to-English translation dataset."""

    def __init__(self):
        """Initialize training and validation data and tokenizers."""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """Create Portuguese and English sub-word tokenizers.

        Args:
            data: A dataset containing Portuguese-English sentence pairs.

        Returns:
            A tuple containing the Portuguese and English tokenizers.
        """
        tokenizer_pt_base = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en_base = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        def portuguese_iterator():
            """Yield Portuguese sentences from the dataset."""
            for pt, _ in data:
                yield pt.numpy().decode('utf-8')

        def english_iterator():
            """Yield English sentences from the dataset."""
            for _, en in data:
                yield en.numpy().decode('utf-8')

        tokenizer_pt = tokenizer_pt_base.train_new_from_iterator(
            portuguese_iterator(),
            vocab_size=2 ** 13
        )
        tokenizer_en = tokenizer_en_base.train_new_from_iterator(
            english_iterator(),
            vocab_size=2 ** 13
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Encode Portuguese and English sentences into token lists.

        Start-of-sentence tokens use the vocabulary size as their index.
        End-of-sentence tokens use the vocabulary size plus one.

        Args:
            pt: A TensorFlow tensor containing a Portuguese sentence.
            en: A TensorFlow tensor containing an English sentence.

        Returns:
            A tuple containing the Portuguese and English token lists.
        """
        pt_sentence = pt.numpy().decode('utf-8')
        en_sentence = en.numpy().decode('utf-8')

        pt_tokens = self.tokenizer_pt.encode(
            pt_sentence,
            add_special_tokens=False
        )
        en_tokens = self.tokenizer_en.encode(
            en_sentence,
            add_special_tokens=False
        )

        pt_start = self.tokenizer_pt.vocab_size
        pt_end = self.tokenizer_pt.vocab_size + 1
        en_start = self.tokenizer_en.vocab_size
        en_end = self.tokenizer_en.vocab_size + 1

        pt_tokens = [pt_start] + pt_tokens + [pt_end]
        en_tokens = [en_start] + en_tokens + [en_end]

        return pt_tokens, en_tokens
