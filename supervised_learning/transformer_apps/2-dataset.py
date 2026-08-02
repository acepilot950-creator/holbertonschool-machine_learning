#!/usr/bin/env python3
"""Dataset preparation module for Portuguese-to-English translation."""

import tensorflow as tf
import transformers
from setup import load_pt2en


class Dataset:
    """Load and prepare a Portuguese-to-English translation dataset."""

    def __init__(self):
        """Initialize, tokenize, and encode the translation datasets."""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """Create Portuguese and English subword tokenizers.

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
        """Encode a translation pair into lists of token IDs.

        Args:
            pt: A tensor containing a Portuguese sentence.
            en: A tensor containing an English sentence.

        Returns:
            A tuple containing Portuguese and English token lists.
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
        pt_end = pt_start + 1
        en_start = self.tokenizer_en.vocab_size
        en_end = en_start + 1

        pt_tokens = [pt_start] + pt_tokens + [pt_end]
        en_tokens = [en_start] + en_tokens + [en_end]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """Wrap the encode method for use in a TensorFlow data pipeline.

        Args:
            pt: A tensor containing a Portuguese sentence.
            en: A tensor containing an English sentence.

        Returns:
            A tuple containing encoded Portuguese and English tensors.
        """
        pt_tokens, en_tokens = tf.py_function(
            self.encode,
            [pt, en],
            [tf.int64, tf.int64]
        )

        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens
