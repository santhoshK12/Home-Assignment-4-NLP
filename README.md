CS5760 – Natural Language Processing
Homework 4 – RNNs, LSTMs, Attention, and Transformers

Student Name: Santhosh Reddy Kistipati
Course: CS5760 – Natural Language Processing
Semester: Fall 2025
Department: Computer Science & Cybersecurity
University: University of Central Missouri
Easily accuss the github link : https://github.com/santhoshK12/Home-Assignment-4-NLP

Overview

This repository contains my complete solutions for Homework 4.
The assignment has two main components:

Short Answer Theoretical Questions

Programming Tasks involving RNNs, LSTMs, and Transformer-based attention.

This README explains:

How each question was solved

Design choices made in the programming section

Outputs produced (loss curves, attention maps, sample generations)

How to run the code

All code files are fully commented as required.

============================================================
PART I — Short Answer Explanations
============================================================

Below are expanded explanations for each theory question. I provide concise definitions followed by deeper clarification wherever relevant.

1. RNN Families & Use-Cases
(a) I/O Pattern Mapping

I matched each NLP task to the suitable RNN input/output pattern based on whether its prediction involves a single output, a sequence of aligned labels, or an output sequence of different length.

Next-word prediction → One-to-Many
The model begins from a single start-of-sequence signal and generates multiple characters or words autoregressively. This fits the one-to-many structure because one input context leads to a multi-step output.

Sentiment of a sentence → Many-to-One
The model must read an entire sequence and compress its meaning into one label. All time steps contribute to a single final classification.

Named Entity Recognition (NER) → Many-to-Many (Aligned)
Each word requires exactly one label. This creates a direct alignment between input tokens and output tags.

Machine translation → Many-to-Many (Unaligned)
Input and output sequence lengths differ. The encoder processes the source sentence, and the decoder generates the target sentence without a token-to-token alignment.

(b) Unrolling & BPTT

Unrolling duplicates the RNN cell across the entire sequence, enabling Backpropagation Through Time (BPTT) to compute gradients across all timesteps while sharing the same weights between every copy.

(c) Advantage & Limitation of Weight Sharing

Advantage: Consistent behavior across positions while reducing parameters and improving generalization.

Limitation: Prevents specialization for specific positions, restricting the model’s ability to independently model distant or position-specific features.

2. Vanishing Gradients & Remedies
(a) Vanishing Gradient Problem

In deep recurrent chains, gradients are repeatedly multiplied by activation derivatives that are less than one. This causes the gradients to shrink exponentially across timesteps, preventing the model from learning long-range dependencies and effectively limiting memory to only recent tokens.

(b) Architectural Remedies

LSTM: Introduces gated memory cells that use additive updates, allowing gradients to flow through the cell state with minimal attenuation.

GRU: Uses reset and update gates to simplify gradient pathways and reduce vanishing effects while requiring fewer parameters than LSTMs.

(c) Training Technique

Gradient Clipping: Keeps gradients within a reasonable range, preventing explosion and maintaining a smoother optimization landscape, which indirectly stabilizes long-range gradient propagation.

3. LSTM Gates & Cell State
(a) Gate Functions and Roles

Forget Gate (sigmoid): Determines which previous cell-state components to remove.

Input Gate (sigmoid + tanh): Selects which new information to write and generates candidate values for the cell state.

Output Gate (sigmoid + tanh): Controls exposure of internal memory by filtering what becomes part of the hidden state.

(b) Linear Gradient Path

The LSTM cell state updates via additive operations and limited nonlinear transformations. This structure preserves gradient magnitude, enabling information to persist through long sequences.

(c) What to Remember vs What to Expose

The forget and input gates manage storage of long-term information, while the output gate determines what portion of this stored information is used at the current timestep.

4. Self-Attention
(a) Definitions

Query: Represents what the current token wants to retrieve.

Key: Represents what information a token contains.

Value: The actual content returned after computing attention weights.

(b) Dot-Product Attention Formula
Attention(Q, K, V) = softmax((QK^T) / sqrt(d_k)) * V

(c) Why Divide by sqrt(d_k)

Normalizing QK^T prevents excessively large logits that lead to extremely peaked softmax outputs, which otherwise produce unstable gradients.

5. Multi-Head Attention & Residuals
(a) Why Multi-Head

Multiple heads enable the model to attend to different semantic or syntactic relationships simultaneously, improving its ability to capture varied patterns in the data.

(b) Purpose of Add & Norm

Residual connections preserve input information and improve gradient flow; LayerNorm maintains stable activation scales, improving convergence and preventing exploding updates.

(c) Example Relation

A head might specialize in coreference (linking pronouns to antecedents) while others capture dependencies like subject–verb agreement.

6. Encoder–Decoder Masked Attention
(a) Purpose of Masking

Masking prevents access to future tokens, enforcing left-to-right prediction and avoiding information leakage during training.

(b) Encoder vs Cross-Attention

Encoder attention uses only the input sequence; cross-attention allows decoder tokens to attend to the encoder’s outputs.

(c) Inference

The decoder predicts one token at a time, appends it to the input, and repeatedly feeds the growing sequence back for the next prediction.

============================================================
PART II — PROGRAMMING SOLUTIONS
============================================================

All code below is written in a student-appropriate format and was run in Google Colab.
Every major block includes comments for clarity, following assignment instructions.

Q1 – Character-Level RNN Language Model

Below is the full implementation used in my notebook. It includes:

Toy corpus and external text

Vocabulary creation

Dataset class

LSTM model

Training loop

Loss plotting

Sampling with temperature

Code Implementation
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

# Optional external text file (data.txt)
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

# Dataset class
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

# Model
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

# Training
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

# Plot loss
plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()

# Sampling
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

Q2 – Mini Transformer Encoder

Includes:

Tokenization

Embeddings

Sinusoidal positional encoding

Multi-head attention

Feed-forward block

Add & Norm

Attention heatmap

Code Implementation
# ===============================================================
# Mini Transformer Encoder for Sentences
# Student: Santhosh Reddy Kistipati
# ===============================================================

import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sentences = [
    "the cat sat on the mat",
    "the dog chased the cat",
    "a bird flew over the house",
    "the mouse ate some cheese",
    "the cat slept all day",
    "the dog barked loudly",
    "the river is near the village",
    "the mouse ran away quickly",
    "the bird is on the tree",
    "the house is very old"
]

tokenized = [s.split() for s in sentences]
vocab = sorted({word for sent in tokenized for word in sent})
vocab = ["<pad>", "<unk>"] + vocab

word2idx = {w:i for i,w in enumerate(vocab)}
idx2word = {i:w for w,i in word2idx.items()}

max_len = max(len(s) for s in tokenized)

def encode(sent):
    ids = [word2idx.get(w,1) for w in sent]
    ids += [0]*(max_len-len(ids))
    return ids[:max_len]

ids = torch.tensor([encode(s) for s in tokenized], device=device)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2)*(-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos*div)
        pe[:, 1::2] = torch.cos(pos*div)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, h):
        super().__init__()
        assert d_model % h == 0
        self.h = h
        self.dk = d_model//h
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x):
        B,L,D = x.size()
        Q = self.Wq(x).view(B,L,self.h,self.dk).transpose(1,2)
        K = self.Wk(x).view(B,L,self.h,self.dk).transpose(1,2)
        V = self.Wv(x).view(B,L,self.h,self.dk).transpose(1,2)
        attn = torch.matmul(Q, K.transpose(-1,-2))/math.sqrt(self.dk)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1,2).contiguous().view(B,L,D)
        return self.Wo(out), attn

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model,h)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.ReLU(),
            nn.Linear(d_ff,d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self,x):
        attn_out, attn = self.attn(x)
        x = self.norm1(x+attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x+ff_out)
        return x, attn

d_model = 64
encoder = TransformerEncoderLayer(d_model,4,128).to(device)
embedding = nn.Embedding(len(vocab),d_model).to(device)
pos = PositionalEncoding(d_model,max_len).to(device)

x = embedding(ids)
x = pos(x)
context, attn_weights = encoder(x)

# Heatmap for sentence 0 head 0
plt.imshow(attn_weights[0,0].cpu().detach().numpy())
plt.xticks(range(max_len), [idx2word[i.item()] for i in ids[0]], rotation=90)
plt.yticks(range(max_len), [idx2word[i.item()] for i in ids[0]])
plt.title("Self-Attention Heatmap (Sentence 0, Head 0)")
plt.colorbar()
plt.tight_layout()
plt.show()

Q3 – Scaled Dot-Product Attention
# ===============================================================
# Scaled Dot-Product Attention
# Student: Santhosh Reddy Kistipati
# ===============================================================

import torch
import torch.nn.functional as F
import math

torch.manual_seed(0)

def attention(Q,K,V):
    d_k = Q.size(-1)
    scores = Q @ K.T
    scaled = scores / math.sqrt(d_k)
    attn = F.softmax(scaled, dim=-1)
    out = attn @ V
    return scores, scaled, attn, out

Q = torch.randn(4,8)
K = torch.randn(4,8)
V = torch.randn(4,8)

raw, scaled, attn, out = attention(Q,K,V)

print("Raw Scores:\n", raw)
print("Scaled Scores:\n", scaled)
print("Attention Weights:\n", attn)
print("Output:\n", out)

print("Softmax on Raw:\n", F.softmax(raw, dim=-1))
print("Softmax on Scaled:\n", F.softmax(scaled, dim=-1))

How to Run

Open Google Colab

Upload the .ipynb file

Run all cells in order

For external text in Q1, upload data.txt

End of README
