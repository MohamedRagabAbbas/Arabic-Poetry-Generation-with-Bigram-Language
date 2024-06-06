# Poem Generation using Bigram Language Model

This project implements a Bigram Language Model using PyTorch, trained on an Arabic text dataset. The model predicts the next word in a sequence based on the previous word. 

## Table of Contents

- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Generating Text](#generating-text)
- [Installation](#installation)
- [Acknowledgements](#acknowledgements)

## Introduction

A bigram model is a type of Markov chain where the probability of each word depends only on the previous word. This project demonstrates how to build, train, and use a bigram model for natural language processing tasks.

## Model Architecture

The model architecture includes:
- An embedding layer to convert words into vector representations.
- A forward pass to compute the logits and the loss.
- A method to generate text by sampling from the model's predictions.

## Training the Model

### Hyperparameters

- `batch_size`: 32
- `block_size`: 8
- `max_iters`: 3000
- `eval_interval`: 300
- `learning_rate`: 0.01
- `eval_iters`: 200

### Training Loop

1. **Batch Generation**: Generate batches of input and target sequences from the training data.
2. **Forward Pass**: Compute the logits and loss.
3. **Backward Pass**: Compute gradients and update model parameters.
4. **Evaluation**: Periodically evaluate the model on the validation set.

## Generating Text

To generate text, start with a given context and iteratively sample the next word based on the model's predictions. Append the sampled word to the context and repeat until the desired length is reached.

## Installation

To run this project, you'll need to install the necessary dependencies. You can use the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### Acknowledgements
This project was developed using the following resources:
- PyTorch
- NLTK
- SpaCy
- Google Colab
