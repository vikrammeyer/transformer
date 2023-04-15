import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    # https://pytorch.org/tutorials/beginner/transformer_tutorial.html#define-the-model
    def __init__(self, dim, max_len=5000):
        super().__init__()
        self.dim = dim

        div_term = torch.exp(torch.arange(0,dim, 2) * (-math.log(10_000) / dim))

        position = torch.arange(max_len).unsqueeze(1) # [max_len, 1]
        pe = torch.zeros((max_len, 1, dim))
        pe[:,0, 0::2] = torch.sin(position * div_term)
        pe[:,0, 1::2] = torch.cos(position * div_term)
        print(pe.shape)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, dim]
        print(x.shape)
        return x + self.pe[:x.size(0)]

class TokenEmbedding(nn.Module):
    def __init__(self, d_emb, n_vocab) -> None:
        super().__init__()

        self.d_emb = d_emb

        self.W = torch.rand((d_emb, n_vocab))


    def forward(self, vocab_idx):
        return self.W[:, vocab_idx]


if __name__ == '__main__':
    pe = PositionalEncoding(10)
    x = torch.rand(15, 32, 10)
    pe(x)