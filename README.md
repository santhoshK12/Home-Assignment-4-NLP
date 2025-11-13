# CS5760 – Natural Language Processing  
## Homework 4 – RNNs, LSTMs, Attention, and Transformers  
**Student Name:** Santhosh Reddy Kistipati  
**Course:** CS5760 – NLP  
**Semester:** Fall 2025  
**Department:** Computer Science & Cybersecurity  
**University:** University of Central Missouri  

---
Easyly accuss the github link : https://github.com/santhoshK12/Home-Assignment-4-NLP/tree/main
## Overview
This repository contains my complete solutions for Homework 4.  
The assignment consists of both theoretical questions and programming tasks involving RNNs, LSTMs, attention mechanisms, and Transformer encoders.

This README explains:
- How each theoretical question was solved  
- How each programming question was implemented  
- Key outputs such as loss curves, generated samples, and attention visualizations  
- Instructions to run the provided notebook  

All source code is commented clearly as required in the assignment instructions.

---

# ================================================
# PART I – Short Answer Explanations
# ================================================

Below are expanded explanations of each short-answer question, written to demonstrate understanding and connect concepts to the lecture slides.

---

## 1. RNN Families & Use-Cases

### (a) I/O Pattern Mapping
Each task was mapped to the appropriate recurrent I/O pattern:

- **Next-word prediction → One-to-Many**  
  A single input (context or start token) produces multiple outputs sequentially. Prediction happens autoregressively.

- **Sentiment classification → Many-to-One**  
  The model reads the entire sentence and compresses its meaning into one sentiment label.

- **Named Entity Recognition (NER) → Many-to-Many (Aligned)**  
  Each input token receives exactly one label, so outputs are aligned step-by-step.

- **Machine Translation → Many-to-Many (Unaligned)**  
  Input and output sequence lengths differ. The encoder processes the source, and the decoder generates the target sentence.

### (b) Unrolling and BPTT
Unrolling duplicates the RNN cell across time so Backpropagation Through Time can compute gradients at each timestep while sharing the same parameters across the entire sequence.

### (c) Weight Sharing Advantage and Limitation
- **Advantage:** Reduces parameters and forces the model to learn generalizable temporal patterns.  
- **Limitation:** Prevents timestep-specific specialization, making it harder to model long-range or position-dependent features.

---

## 2. Vanishing Gradients & Remedies

### (a) Vanishing Gradient Problem
In RNNs, gradients shrink exponentially when propagated through many nonlinear timesteps. This prevents the network from learning long-range dependencies, causing it to focus mostly on recent inputs.

### (b) Architectural Remedies
- **LSTM:** Introduces gated memory and additive updates, creating a stable gradient path.  
- **GRU:** Simplifies gating and reduces vanishing by passing gradients through fewer transformations.

### (c) Training Technique
**Gradient clipping** stabilizes training by preventing exploding gradients, which indirectly keeps the optimization landscape smooth and prevents deeper vanishing issues.

---

## 3. LSTM Gates & Cell State

### (a) Roles of Gates
- **Forget Gate (sigmoid):** Controls which previous cell-state components are discarded.  
- **Input Gate (sigmoid + tanh):** Selects and prepares new information for writing into memory.  
- **Output Gate (sigmoid + tanh):** Determines what information from the memory is exposed to the hidden state.

### (b) Linear Path for Gradients
Because the LSTM cell uses additive updates instead of repeated multiplications, gradients can flow more stably and avoid vanishing over long sequences.

### (c) Remember vs. Expose  
The forget and input gates determine what gets stored for long-term memory, while the output gate determines what part of that stored information is revealed at each timestep.

---

## 4. Self-Attention

### (a) Definitions
- **Query:** Represents what information the token is seeking.  
- **Key:** Represents what information a token contains.  
- **Value:** The content passed forward once attention scores are computed.

### (b) Dot-Product Attention Formula
```
Attention(Q, K, V) = softmax((QK^T) / sqrt(d_k)) * V
```

### (c) Why Divide by sqrt(d_k)
Scaling prevents extremely large dot products that cause sharp, unstable softmax distributions and poor gradients.

---

## 5. Multi-Head Attention & Residuals

### (a) Reason for Multi-Head Attention
Multiple heads let the model attend to multiple types of relationships simultaneously (syntactic, semantic, positional), making the representation richer than a single head.

### (b) Purpose of Add & Norm
Residual connections preserve original information and improve gradient flow. LayerNorm stabilizes activation ranges, making training more reliable.

### (c) Example Relation
A head may capture pronoun resolution (coreference), while another may capture syntactic dependencies such as subjects linked to verbs.

---

## 6. Encoder–Decoder Masked Attention

### (a) Purpose of Masked Self-Attention
Masking blocks the decoder from attending to future tokens, preventing information leakage during training and enforcing autoregressive prediction.

### (b) Encoder vs Cross-Attention
Encoder attention processes only the input sequence; cross-attention allows decoder tokens to attend to encoder outputs when generating translations.

### (c) Inference Process
During inference, the decoder generates one token at a time, appends it, and feeds the entire sequence back to predict the next token until completion.

---

# ================================================
# PART II – PROGRAMMING SOLUTIONS
# ================================================

All implementations are included inside the notebook file.  
Below is the complete source code for each programming question.

---

# Q1 – Character-Level RNN Language Model (LSTM)

````python
# ===============================================================
# Character-Level RNN Language Model (LSTM version)
# Student: Santhosh Reddy Kistipati
# ===============================================================

import os
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Toy corpus
toy_text = (
    "hello hello help hell hello helpful help " * 50 +
    "this is a small example corpus for a character level rnn. "
)

TEXT_PATH = "/content/data.txt"
if os.path.exists(TEXT_PATH):
    with open(TEXT_PATH, "r", encoding="utf-8") as f:
        extra = f.read()
    text = toy_text + extra.lower()
else:
    text = toy_text.lower()

chars = sorted(list(set(text)))
vocab_size = len(chars)

char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for ch, i in char2idx.items()}

data = torch.tensor([char2idx[c] for c in text], dtype=torch.long)

class CharDataset(Dataset):
    def __init__(self, data, seq_len=100):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+1:idx+1+self.seq_len]
        return x, y

seq_len = 100
batch_size = 64
embed_dim = 64
hidden = 128

dataset = CharDataset(data, seq_len)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x, h=None):
        x = self.embed(x)
        out, h = self.lstm(x, h)
        logits = self.fc(out)
        return logits, h

model = CharLSTM(vocab_size, embed_dim, hidden).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

epochs = 10
train_losses = []

for ep in range(epochs):
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits, _ = model(x)
        loss = crit(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
    train_losses.append(total_loss)
    print(f"Epoch {ep+1}, Loss: {total_loss:.4f}")

plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()

def sample(model, start="h", length=300, temp=1.0):
    model.eval()
    chars_out = [start]
    idx = torch.tensor([[char2idx[start]]], device=device)
    h = None
    for _ in range(length):
        logits, h = model(idx, h)
        logits = logits[:, -1, :] / temp
        prob = torch.softmax(logits, dim=-1)
        idx = torch.multinomial(prob, 1)
        chars_out.append(idx2char[idx.item()])
    return "".join(chars_out)

for t in [0.7, 1.0, 1.2]:
    print(f"Temperature {t}")
    print(sample(model, temp=t))
