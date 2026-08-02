# QA Bot

This project focuses on building a question-answering bot using pre-trained natural language processing models. The bot extracts an answer to a given question from a reference document.

## Learning Objectives

By the end of this project, I should be able to explain:

* What Question Answering is
* How extractive question answering works
* How BERT can be used for question answering
* How to tokenize a question and a reference document
* How to use pre-trained models from TensorFlow Hub
* How to use a pre-trained tokenizer from the Transformers library
* How to identify the start and end positions of an answer
* How to convert BERT tokens back into readable text

## Requirements

### General

* Allowed editors: `vi`, `vim`, `emacs`
* All files are interpreted or compiled on Ubuntu 20.04 LTS
* Python version: `3.9`
* NumPy version: `1.25.2`
* TensorFlow version: `2.15`
* TensorFlow Hub version: `0.15.0`
* Transformers version: `4.44.2`
* All files must end with a new line
* The first line of every Python file must be:

```python
#!/usr/bin/env python3
```

* All Python files must be executable
* Code must follow `pycodestyle` version `2.11.1`
* All modules, classes, and functions must contain documentation

## Installation

Install TensorFlow Hub:

```bash
pip install --user tensorflow-hub==0.15.0
```

Install Transformers:

```bash
pip install --user transformers==4.44.2
```

## Dataset

The project uses a collection of Holberton USA Zendesk articles contained in:

```text
ZendeskArticles.zip
```

After extraction, the articles are located in the `ZendeskArticles` directory.

## Project Files

| File        | Description                                                                                                  |
| ----------- | ------------------------------------------------------------------------------------------------------------ |
| `0-qa.py`   | Extracts an answer to a question from a reference document using a pre-trained BERT question-answering model |
| `README.md` | Contains information about the project and its requirements                                                  |

## Tasks

### 0. Question Answering

The `question_answer` function finds a snippet of text inside a reference document that answers a given question.

Prototype:

```python
def question_answer(question, reference):
```

Arguments:

* `question`: a string containing the question
* `reference`: a string containing the document in which the answer should be found

Return value:

* A string containing the extracted answer
* `None` when no valid answer is found

The function uses:

* The `bert-uncased-tf2-qa` model from TensorFlow Hub
* The `bert-large-uncased-whole-word-masking-finetuned-squad` tokenizer from the Transformers library

## Usage

Example:

```python
#!/usr/bin/env python3

question_answer = __import__('0-qa').question_answer

with open('ZendeskArticles/PeerLearningDays.md') as file:
    reference = file.read()

question = 'When are PLDs?'
answer = question_answer(question, reference)

print(answer)
```

Expected output:

```text
on - site days from 9 : 00 am to 3 : 00 pm
```

## How It Works

The question and reference document are tokenized into the following general structure:

```text
[CLS] question [SEP] reference [SEP]
```

The BERT question-answering model produces:

* Scores for the possible starting token of the answer
* Scores for the possible ending token of the answer

The tokens between the selected start and end positions are converted back into a readable string.

## Style and Documentation Checks

Check the code style with:

```bash
pycodestyle 0-qa.py
```

Check the module documentation with:

```bash
python3 -c 'print(__import__("0-qa").__doc__)'
```

Check the function documentation with:

```bash
python3 -c \
'print(__import__("0-qa").question_answer.__doc__)'
```

Make the Python file executable:

```bash
chmod +x 0-qa.py
```
