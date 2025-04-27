import torch
import math
from torch import nn


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        """
        입력 임베딩을 정의하는 클래스
        :param d_model: model의 차원 수
        :param vocab_size: 들어오는 단어 개수
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(seq_len)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model) # seq_len * d_model 의 0으로 채운 행렬 반환

        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) #0부터 seq_len까지 float 형식의 일정한 간격 센서 생성
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, Seq_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
        """
        epsilon의 목적 : 1) CPU,GPU 연산 시 너무 큰 값 / 작은 값 방지를 위해, 2) DivByZero 방지를 위해
        """
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # Added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias