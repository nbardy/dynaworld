import pytest
import torch
import torch.nn as nn

from gs_models.dynamic_video_token_gs_implicit_camera import QueryCrossAttentionBlock, _manual_batch_first_mha


def test_manual_batch_first_mha_matches_torch_mha_forward_and_gradients():
    torch.manual_seed(4)
    mha = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True, dropout=0.0)
    query = torch.randn(2, 5, 16, requires_grad=True)
    memory = torch.randn(2, 11, 16, requires_grad=True)
    value = torch.randn(2, 11, 16, requires_grad=True)

    expected, _ = mha(query, memory, value, need_weights=False)
    actual = _manual_batch_first_mha(mha, query, memory, value)
    assert torch.allclose(actual, expected, atol=1.0e-6, rtol=1.0e-5)

    grad_seed = torch.randn_like(expected)
    expected_grads = torch.autograd.grad(
        expected,
        (query, memory, value, *tuple(mha.parameters())),
        grad_outputs=grad_seed,
        retain_graph=True,
    )
    actual_grads = torch.autograd.grad(
        actual,
        (query, memory, value, *tuple(mha.parameters())),
        grad_outputs=grad_seed,
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert torch.allclose(actual_grad, expected_grad, atol=1.0e-6, rtol=1.0e-5)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is required for the large-memory smoke.")
def test_query_cross_attention_large_memory_smoke_on_mps():
    torch.manual_seed(5)
    block = QueryCrossAttentionBlock(dim=64, num_heads=4, mlp_ratio=1.0).to("mps")
    queries = torch.randn(1, 10, 64, device="mps", requires_grad=True)
    memory = torch.randn(1, 40960, 64, device="mps", requires_grad=True)

    output = block(queries, memory)
    loss = output.square().mean()
    loss.backward()
    torch.mps.synchronize()

    assert output.shape == queries.shape
    assert torch.isfinite(output).all().item()
    assert queries.grad is not None
    assert memory.grad is not None
