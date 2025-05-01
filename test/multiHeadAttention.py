import torch
import torch.nn as nn
import math

from model import MultiHeadAttentionBlock  # 필요 시 import

def test_multihead_attention():
    # 테스트 설정
    batch_size = 2
    seq_len = 5
    d_model = 512
    num_heads = 8
    dropout = 0.1

    # 가짜 입력 생성 (batch_size, seq_len, d_model)
    x = torch.randn(batch_size, seq_len, d_model)

    # 마스크 (필요 없으면 None 가능)
    mask = torch.ones(batch_size, 1, 1, seq_len)  # broadcast 가능 형태 (B, 1, 1, L)

    # 모델 인스턴스
    mha = MultiHeadAttentionBlock(d_model, num_heads, dropout)

    # Forward 패스 (Q=K=V)
    out = mha(x, x, x, mask)

    # 결과 확인
    print("✅ Output shape:", out.shape)  # (batch_size, seq_len, d_model)
    assert out.shape == (batch_size, seq_len, d_model), "Output shape mismatch"

    print("✅ Attention scores shape:", mha.attention_scores.shape)  # (B, h, L, L)
    assert mha.attention_scores.shape == (batch_size, num_heads, seq_len, seq_len), "Attention scores shape mismatch"

    print("🎉 테스트 통과")


def test_attention_embedding_change():
    batch_size = 2
    seq_len = 5
    d_model = 512
    num_heads = 8
    dropout = 0.0  # 차이 분석할 때는 dropout 없이

    # 가짜 입력 (Q = K = V)
    x = torch.randn(batch_size, seq_len, d_model)

    # 모델 인스턴스
    mha = MultiHeadAttentionBlock(d_model, num_heads, dropout)

    # Forward 수행
    output = mha(x, x, x, mask=None)

    # 1. Shape 체크
    assert output.shape == x.shape, "Shape mismatch"

    # 2. 임베딩 변화 확인 (예: Cosine Similarity / Norm 차이)
    cosine_sim = nn.functional.cosine_similarity(x, output, dim=-1)  # (B, L)
    l2_diff = torch.norm(x - output, dim=-1)                         # (B, L)

    print("📐 평균 Cosine Similarity (입력 vs 출력):", cosine_sim.mean().item())
    print("📏 평균 L2 차이 (입력 vs 출력):", l2_diff.mean().item())

    # 3. 시각적 결과
    for b in range(batch_size):
        for i in range(seq_len):
            print(f"Token {i+1} in batch {b+1}:")
            print(f"  CosSim = {cosine_sim[b, i]:.4f}, L2 Diff = {l2_diff[b, i]:.4f}")

# 실행
test_multihead_attention()
test_attention_embedding_change()