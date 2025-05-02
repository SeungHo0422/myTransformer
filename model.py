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
        pe = torch.zeros(seq_len, d_model)  # seq_len * d_model 의 0으로 채운 행렬 반환

        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # 0부터 seq_len까지 float 형식의 일정한 간격 센서 생성
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, Seq_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10 ** -6) -> None:
        """
        epsilon의 목적 : 1) CPU,GPU 연산 시 너무 큰 값 / 작은 값 방지를 위해, 2) DivByZero 방지를 위해
        """
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # Multiplied
        self.bias = nn.Parameter(torch.zeros(1))  # Added

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.linear(d_model, d_ff)  # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.linear(d_ff, d_model)  # W2 and B2

    def forward(self, x):
        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_ff) --> (Batch, Seq_Len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # 입력 임베딩 차원 수
        self.h = h  # 헤드 수
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)  # WQ
        self.w_k = nn.Linear(d_model, d_model)  # WK
        self.w_v = nn.Linear(d_model, d_model)  # WV

        self.w_o = nn.Linear(d_model, d_model)  # WO
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]  # 현재 헤드에서의 임베딩 차원 수

        # (Batch, h, Seq_Len, d_k) --> (Batch, h, Seq_Len, Seq_Len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)  # (Batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores  # (Batch, h, seq_len, seq_len) --> (Batch, h, Seq_Len, d_k)

    def forward(self, q, k, v, mask):
        query = self.w_q(q)  # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_model)
        key = self.w_k(k)  # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_model)
        value = self.w_v(v)  # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_model)

        # (Batch, Seq_Len, d_model) --[MultiHead를 위해 d_model을 h*d_k로 쪼갠다고 생각하자.]--> (Batch, Seq_Len, h, d_k) --> (Batch, h, Seq_len, d_k)[이렇게 바꿔야 각 head별로 attention 수행]
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        """
        x.transpose(1, 2) : Attention 계산 결과는 각 head마다 따로 있음. 따라서, 합치기 위해 (Batch, Seq_Len, h, d_k) 구조로 정렬할 필요가 있음.
        .contiguous() : .view를 안전하게 쓰기 위한 Tensor 연속성 보장 (메모리 재배치)
        .view() : h개의 head 추력을 이어붙여 하나의 벡터로 복원. 결과적으로 (Batch, Seq_Len, d_model) 형태 차원으로 복원
        """
        # (Batch, h, Seq_Len, d_k) --> (Batch, Seq_Len, h, d_k) --> (Batch, Seq_Len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[2], self.h * self.d_k)

        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        return self.w_o(x)

class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

