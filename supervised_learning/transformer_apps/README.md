# Transformer Applications

## Description

This project explores practical applications of Transformer models using TensorFlow and the Hugging Face Transformers library.

The main objective is to build a Portuguese-to-English machine translation pipeline using the `ted_hrlr_translate/pt_to_en` dataset. The project includes dataset loading, tokenizer training, data preprocessing, Transformer model preparation, training, and translation.

The Portuguese and English texts are tokenized using BERT-based tokenizers:

* `neuralmind/bert-base-portuguese-cased` for Portuguese
* `bert-base-uncased` for English

The tokenizers are retrained on the project's training dataset with a maximum vocabulary size of `2**13`, or 8192 tokens.

## Learning Objectives

By the end of this project, you should be able to explain:

* What a Transformer is
* How Transformers are used in natural language processing
* How machine translation datasets are prepared
* What subword tokenization is
* How pretrained tokenizers can be adapted to a new dataset
* How to use Hugging Face tokenizers
* How to prepare TensorFlow datasets for Transformer models
* How encoder-decoder Transformer models perform machine translation

## Dataset

This project uses the Portuguese-to-English subset of the TED Talks translation dataset:

```text
ted_hrlr_translate/pt_to_en
```

The original TensorFlow Datasets source is no longer available. Therefore, the dataset is downloaded from a Holberton-hosted mirror and loaded using the provided `setup.py` helper.

The dataset contains Portuguese and English sentence pairs represented as:

```python
(portuguese_sentence, english_sentence)
```

Available splits:

| Split      | Number of sentence pairs |
| ---------- | -----------------------: |
| Training   |                   51,785 |
| Validation |                    1,193 |
| Test       |                    1,803 |

## Dataset Setup

Download the dataset archive:

```bash
curl -L -O \
https://holbucket-prod.s3.fr-par.scw.cloud/projects/2422/ted_hrlr_pt_to_en.tar.gz
```

Create the dataset cache directory:

```bash
mkdir -p ~/.cache/ted_hrlr
```

Extract the dataset:

```bash
tar -xzvf ted_hrlr_pt_to_en.tar.gz -C ~/.cache/ted_hrlr
```

Download the dataset helper into the root of the project:

```bash
curl -L -O \
https://holbucket-prod.s3.fr-par.scw.cloud/projects/2422/setup.py
```

Verify that the dataset is installed correctly:

```bash
python3 setup.py
```

The expected output includes:

```text
train: 51785 pairs
validation: 1193 pairs
test: 1803 pairs
```

## Installation

Install the required versions of the project dependencies:

```bash
pip install --user tensorflow-datasets==4.9.2
pip install --user transformers==4.44.2
```

The project is tested with:

* Ubuntu 20.04 LTS
* Python 3.9
* NumPy 1.25.2
* TensorFlow 2.15
* Transformers 4.44.2
* pycodestyle 2.11.1

## Files

### `0-dataset.py`

Contains the `Dataset` class, which loads and prepares the Portuguese-to-English translation dataset.

The class creates the following public instance attributes:

* `data_train`: the training split loaded with `load_pt2en('train')`
* `data_valid`: the validation split loaded with `load_pt2en('validation')`
* `tokenizer_pt`: the tokenizer trained on Portuguese sentences
* `tokenizer_en`: the tokenizer trained on English sentences

The tokenizers are created by the `tokenize_dataset` method.

### `setup.py`

Provides the `load_pt2en` function used to load the local dataset splits as TensorFlow datasets.

## Tokenization

The project uses pretrained BERT tokenizers as the basis for new subword tokenizers.

Portuguese tokenizer:

```python
transformers.AutoTokenizer.from_pretrained(
    'neuralmind/bert-base-portuguese-cased'
)
```

English tokenizer:

```python
transformers.AutoTokenizer.from_pretrained(
    'bert-base-uncased'
)
```

Each tokenizer is retrained on the corresponding language from the training split:

```python
tokenizer.train_new_from_iterator(
    text_iterator,
    vocab_size=2 ** 13
)
```

Only the training split is used to train the tokenizers. The validation split is not used for tokenizer training.

## Usage

Make the Python files executable:

```bash
chmod +x 0-dataset.py
chmod +x 0-main.py
```

Run the test file:

```bash
./0-main.py
```

Example output:

```text
entre todas as grandes privações com que nos debatemos hoje ...
amongst all the troubling deficits we struggle with today ...
isso corresponde ao dobro do tempo da existência dos homens neste planeta .
that's twice as long as humans have been on this planet .
<class 'transformers.models.bert.tokenization_bert_fast.BertTokenizerFast'>
<class 'transformers.models.bert.tokenization_bert_fast.BertTokenizerFast'>
```

## Code Style

All Python files:

* Start with `#!/usr/bin/env python3`
* End with a new line
* Follow `pycodestyle` version 2.11.1
* Include documentation for modules, classes, and functions
* Are executable

Check the code style with:

```bash
pycodestyle 0-dataset.py
```

## Documentation Checks

Check the module documentation:

```bash
python3 -c "print(__import__('0-dataset').__doc__)"
```

Check the class documentation:

```bash
python3 -c "print(__import__('0-dataset').Dataset.__doc__)"
```

Check the method documentation:

```bash
python3 -c \
"print(__import__('0-dataset').Dataset.tokenize_dataset.__doc__)"
```


