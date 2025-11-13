# CS5760 – Natural Language Processing  
## Homework 4 – RNNs, LSTMs, Attention, and Transformers  

**Student Name:** Santhosh Reddy Kistipati  
**Course:** CS5760 – NLP  
**Semester:** Fall 2025  
**Department:** Computer Science & Cybersecurity  
**University:** University of Central Missouri  
easyly access the all data throw the Github link : https://github.com/santhoshK12/Home-Assignment-4-NLP/tree/main

---

## Overview

This repository contains my complete solutions for Homework 4.

The assignment has two components:

1. **Part I – Short Answer Questions**  
2. **Part II – Programming Tasks** using RNNs, LSTMs, self-attention, and scaled dot-product attention.

This README explains, in words, how I implemented and structured the three programming questions (Q1, Q2, Q3) without including direct source code. It is intended as a high-level technical explanation for how the code works and how it connects to the course concepts.

All code in the notebook is appropriately commented to reflect these design choices.

---

# ================================================
# PART I – Short Answer (Summary Only)
# ================================================

I answered six theory questions covering:

- Mapping NLP tasks to RNN families (one-to-many, many-to-one, many-to-many aligned/unaligned).  
- Explaining vanishing gradients and remedies (LSTMs, GRUs, gradient clipping).  
- Detailing LSTM gate roles and the cell state as a linear gradient path.  
- Defining Query, Key, and Value in self-attention and writing the dot-product attention formula.  
- Motivating multi-head attention and Add & Norm in Transformers.  
- Explaining masked self-attention in encoder–decoder models and how inference generates tokens step by step.

These answers are written as short, focused paragraphs for each sub-question and stored in the PDF document.

---

# ================================================
# PART II – PROGRAMMING QUESTIONS
# ================================================

Below I describe how I implemented each programming task in detail, from data preparation to model design, training, and outputs.

---

## Q1 – Character-Level RNN Language Model (“hello” Toy & Beyond)

### Goal

The goal of Q1 is to build a character-level language model that predicts the next character given previous characters. The model is trained on a toy corpus and then extended to a larger text file (public-domain style). The architecture is:

> Embedding → RNN (LSTM) → Linear → Softmax over characters

### Data Preparation

I followed the assignment instructions to construct the dataset in two stages:

1. **Toy Corpus:**  
   I started with a small, manually created text consisting of repeated patterns like “hello”, “help”, “hell”, “helpful”, etc. This helps verify that the model is working on a simple controlled example.

2. **Extended Corpus:**  
   The code then checks for a larger text file (about 50–200 KB) in the environment. If found, it appends this text to the toy corpus. This creates a more realistic training setting and tests the model’s ability to generalize beyond the toy pattern.

For both stages:

- All characters are converted to lowercase to simplify the vocabulary.
- I build a character vocabulary by taking the unique set of characters appearing in the text.
- I map each character to an integer index and convert the entire text to a sequence of indices.
- Then I create training examples using a sliding window of fixed sequence length (e.g., 100 characters).  
  - Input sequence: characters at positions `[t, ..., t + L - 1]`  
  - Target sequence: characters at positions `[t + 1, ..., t + L]`  
  This naturally sets up the next-character prediction task.

### Model Architecture

I implemented a simple LSTM-based language model at the character level:

1. **Embedding Layer:**  
   Converts each character index into a dense vector of fixed dimension (e.g., 64). This acts like a learned character embedding.

2. **LSTM Layer:**  
   - Input: sequence of embeddings of shape `(batch_size, seq_len, embed_dim)`  
   - It processes the sequence and maintains hidden and cell states across timesteps, modeling temporal dependencies between characters.
   - The hidden size is chosen in the assignment range (e.g., 128), which balances model capacity and training time.

3. **Linear Output Layer:**  
   For each timestep, the LSTM’s hidden state is projected to a vector of size equal to the vocabulary size (number of characters). These logits represent unnormalized scores for each possible next character.

4. **Softmax (Implicit in Loss):**  
   During training, I use the cross-entropy loss, which internally applies softmax to convert logits into probabilities over the character vocabulary.

### Training Procedure

- **Teacher Forcing:**  
  During training, the model always receives the true previous character as input at the next timestep, not its own sampled prediction. This is implemented by feeding the full ground-truth sequence and shifting it by one position for targets.

- **Loss Function:**  
  I use cross-entropy loss across all positions in the sequence. The model predicts the probability of the next character at each timestep, and the loss is averaged (or summed) across the batch and sequence.

- **Optimizer:**  
  Adam optimizer is used with a small learning rate (e.g., 0.001). This helps with stable convergence on this small dataset.

- **Gradient Clipping:**  
  Before each optimizer step, gradients are clipped to a maximal norm (e.g., 1.0) to prevent exploding gradients, which is common in RNN training.

- **Epochs:**  
  I train the model for 5–20 epochs, as specified. In my experiments, around 10 epochs were enough for the toy dataset to show learning progress.

The code records the total training loss per epoch and prints it to monitor convergence. For an extended version, a validation split could be used similarly to Q2, but Q1 primarily focuses on training and sampling.

### Loss Curves and Sampling

1. **Loss Curve:**  
   I plot training loss versus epoch. A downward trend indicates that the model is learning to predict characters more accurately.

2. **Temperature Sampling:**  
   After training, I implement a sampling loop:
   - Start with an initial character (e.g., `'h'` or a small prefix).
   - At each step:
     - Use the model to compute logits for the next character.
     - Divide logits by a temperature `τ`.  
       - Lower `τ` (e.g., 0.7) yields more conservative, high-probability choices.  
       - Higher `τ` (e.g., 1.2) yields more random and diverse output.
     - Apply softmax to get a probability distribution over characters.
     - Randomly sample the next character from this distribution.
   - Feed the sampled character back into the model and repeat for 200–400 characters.

I generate three sequences for temperatures `0.7`, `1.0`, and `1.2` to show how temperature affects creativity versus coherence.

### Reflection (Conceptual)

In the written report section, I describe how:

- Increasing **sequence length** gives the model more context but makes optimization harder and more memory-intensive.
- Increasing **hidden size** increases capacity (and potentially quality) but also risk of overfitting and slower training.
- Changing **temperature** modifies the balance between repetitive but coherent text (low temperature) and creative but possibly nonsensical text (high temperature).

---

## Q2 – Mini Transformer Encoder for Sentences

### Goal

The goal of Q2 is to implement a minimal version of a Transformer encoder (not the full encoder–decoder), apply it to a batch of short sentences, and visualize self-attention.

Key components to implement:

- Tokenization and embeddings  
- Sinusoidal positional encoding  
- Multi-head self-attention  
- Feed-forward network  
- Add & Norm (residual connections + LayerNorm)  
- Visualization of attention heatmaps between words

### Dataset and Tokenization

I created a small dataset of 10 simple English sentences involving cats, dogs, mice, birds, houses, and rivers. These sentences are short and share vocabulary, making them suitable for a toy Transformer.

Steps:

1. Split each sentence on whitespace into tokens.
2. Build a vocabulary of all unique words across the dataset.
3. Add special tokens `<pad>` and `<unk>` to handle padding and unknown words.
4. Map each word to an integer ID.
5. Pad all sentences to the same maximum length with `<pad>` so that the input has shape `(batch_size, max_len)`.

This produces a batch of token IDs suitable as input to an embedding layer.

### Embeddings and Positional Encoding

1. **Embedding Layer:**  
   Each token ID is mapped to a dense embedding vector of dimension `d_model` (e.g., 64). This creates a tensor of shape `(batch_size, seq_len, d_model)`.

2. **Sinusoidal Positional Encoding:**  
   I implemented the standard sinusoidal positional encoding from the original Transformer paper:
   - For each position and each dimension, use sine and cosine functions with different frequencies.
   - Add this positional encoding tensor to the token embeddings.
   This allows the model to encode word order without relying on recurrence.

### Multi-Head Self-Attention

I implemented multi-head self-attention in the standard way:

1. **Linear Projections:**  
   For each token embedding, I compute:
   - Query (Q)  
   - Key (K)  
   - Value (V)  
   using learned linear layers. Each of these has dimension `d_model`.

2. **Head Splitting:**  
   The Q, K, V tensors are reshaped to split into `h` heads, where each head has dimension `d_k = d_model / h`. This creates tensors of shape `(batch_size, num_heads, seq_len, d_k)`.

3. **Scaled Dot-Product Attention:**  
   For each head:
   - Compute attention scores via `Q * K^T / sqrt(d_k)` for all pairs of tokens.
   - Apply softmax along the key dimension to obtain attention weights.
   - Multiply the attention weights by V to get an attention output per head.

4. **Concatenation and Output Projection:**  
   The outputs of all heads are concatenated along the feature dimension and passed through a final linear layer to project back to dimension `d_model`.

The self-attention mechanism allows each word in a sentence to attend to every other word, including itself, capturing global dependencies in a single layer.

### Feed-Forward Network and Add & Norm

For the encoder layer:

1. **First Residual Block (Attention + Add & Norm):**
   - Input: token representations with positional encoding.
   - Apply multi-head self-attention to obtain attention outputs.
   - Add the attention output back to the original input (residual connection).
   - Apply Layer Normalization to stabilize training.

2. **Second Residual Block (Feed-Forward + Add & Norm):**
   - Apply a position-wise feed-forward network, consisting of:
     - A linear layer expanding the dimension from `d_model` to `d_ff` (e.g., 128).
     - A ReLU activation.
     - A linear layer projecting back from `d_ff` to `d_model`.
   - Add this output back to the input of this block.
   - Apply another LayerNorm.

The result is a single Transformer encoder layer that produces contextualized embeddings for each token.

### Outputs and Attention Heatmap

After running the batch through the encoder:

- I print the tokenized input sentences and their corresponding token IDs.
- I inspect the shape of the final contextual embeddings to verify it is `(batch_size, seq_len, d_model)`.
- I extract the attention weights for a specific sentence (e.g., sentence 0) and a specific head (e.g., head 0), which results in a matrix of shape `(seq_len, seq_len)`.

To visualize attention:

- I plot this attention matrix as a heatmap using Matplotlib.
- Word tokens are used as labels on both axes.
- This makes it easy to see which words attend strongly to which other words in the sentence.

This directly connects to the slides’ explanation of how self-attention helps capture relationships like subject–verb, object, and contextual cues.

---

## Q3 – Scaled Dot-Product Attention

### Goal

The goal of Q3 is to implement the core attention computation from the Transformer:

> Attention(Q, K, V) = softmax((QKᵀ) / sqrt(d_k)) V

and to test it on random inputs while checking softmax stability.

### Function Design

The attention function I implemented takes as input three matrices:

- `Q` (Queries) of shape `(seq_len_q, d_k)`  
- `K` (Keys) of shape `(seq_len_k, d_k)`  
- `V` (Values) of shape `(seq_len_k, d_v)`

The steps inside the function are:

1. **Dot-Product Scores:**  
   Compute unscaled attention scores by multiplying Q with the transpose of K. This yields a matrix of shape `(seq_len_q, seq_len_k)`.

2. **Scaling by sqrt(d_k):**  
   Divide the scores by the square root of the key dimension `d_k`. This is the scaled dot-product part and is critical for numerical stability and gradient behavior.

3. **Softmax over Keys:**  
   Apply softmax along the key dimension to convert scores into attention weights. Each row now represents how much each query attends to every key.

4. **Weighted Sum of Values:**  
   Multiply the attention weight matrix with V to get the final output of shape `(seq_len_q, d_v)`. This represents a weighted combination of value vectors for each query position.

The function returns:

- Unscaled scores  
- Scaled scores  
- Attention weights  
- Output vectors  

This separation is helpful for debugging and for the softmax stability check.

### Testing with Random Q, K, V

To test the implementation:

1. Generate random Q, K, and V matrices using a fixed random seed for reproducibility.
2. Pass them through the attention function.
3. Print:
   - The raw scores (QKᵀ) before scaling  
   - The scaled scores (QKᵀ / sqrt(d_k))  
   - The attention weight matrix (after softmax)  
   - The output vectors

### Softmax Stability Check

To demonstrate why scaling is necessary:

- I compute softmax on the **unscaled scores** and on the **scaled scores**.
- I print both sets of softmax outputs and compare them.
- Typically, the unscaled softmax is more “peaked” and sensitive to small changes in logits, whereas the scaled version is smoother and more stable.

I also print some simple statistics of the scores (mean, standard deviation, min, max) before and after scaling to show numerically how scaling compresses the range of logits.

This small, isolated experiment directly connects to the lecture explanation of scaled dot-product attention and offers a concrete view of how scaling affects the softmax distribution.

---

# How to Run the Code

1. Open the notebook file `Homework_4_NLP_programming_.ipynb` in Google Colab or a local Jupyter environment.  
2. Make sure `PyTorch` and `matplotlib` are available (Colab has them preinstalled).  
3. Run all cells from top to bottom:
   - Q1: Character-level RNN language model.  
   - Q2: Mini Transformer encoder and attention heatmap.  
   - Q3: Scaled dot-product attention demonstration.  
4. Optionally, upload a custom plain-text file for Q1 to expand the corpus.

---

# Student Information

**Name:** Santhosh Reddy Kistipati  
**Course:** CS5760 – Natural Language Processing  
**Semester:** Fall 2025  

This README summarizes both the design and the implementation choices made for Homework 4 and is meant to accompany the fully commented source code in this repository.
