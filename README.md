# Sentence Transformer with Multi-Task Learning

## Overview
This repository contains the implementation of a Sentence Transformer model capable of producing fixed-length embeddings from input sentences. The model is further expanded to handle multi-task learning, focusing on Sentence Classification and an additional task (e.g., Sentiment Analysis). 
The project demonstrates how to structure a neural network to handle multiple NLP tasks simultaneously without training the model, focusing instead on the architecture and forward pass.

## Objectives
- Implement a Sentence Transformer model.
- Extend the model architecture to support multi-task learning with two distinct NLP tasks.
- Discuss architectural choices and strategies for managing multi-task learning.

## Installation

### Prerequisites
- Python 3.10
- Pip

### Setup and Installation
Clone the repository and install the required packages:

```bash
git clone https://github.com/MahithaSann/Sentence-Transformers.git
cd Sentence-Transformers
python -m venv venv
source venv/bin/activate  # Use `.\venv\Scripts\activate` on Windows
pip install -r requirements.txt #takes a few minutes

python .\sentence-transformer-1.py
python .\sentence-transformer-2.py