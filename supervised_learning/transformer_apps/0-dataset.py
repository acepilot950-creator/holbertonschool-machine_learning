#!/usr/bin/env python3
"""Dataset preparation module for Portuguese-to-English translation."""

import transformers
from setup import load_pt2en


class Dataset:
    """Load and prepare a Portuguese-to-English translation dataset."""

    def __init__(self):
        """Initialize training, validation datasets, and tokenizers."""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """Create Portuguese and English sub-word tokenizers.

        Args:
            data: A tf.data.Dataset containing Portuguese-English pairs.

        Returns:
            A tuple containing the Portuguese tokenizer and the English
            tokenizer.
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
