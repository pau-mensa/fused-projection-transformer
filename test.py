import pytest
import torch
import torch.nn as nn

from modules import (
    BaselineProjCE,
    FusedProjCE,
)

def f(module: nn.Module, x: torch.Tensor, y: torch.Tensor):
    
    if isinstance(module, FusedProjCE):
        loss = module(x.view(-1, x.size(-1)), y.view(-1))
        loss.backward()
        proj_grad = module.w_proj.grad
    else:
        loss = module(x, y)
        loss.backward()
        proj_grad = module.w_proj.weight.grad
        
    grad = x.grad
    
    return loss, grad, proj_grad

@pytest.mark.parametrize("seq_len", [256, 1536])
@pytest.mark.parametrize("vocab_size", [2048, 8192])
@pytest.mark.parametrize("hidden_dim", [64, 512])
def test_correctness(
    seq_len, vocab_size, hidden_dim
):
    torch.manual_seed(0)
    device = 'cuda'
    dtype = torch.float32

    x = torch.randn((4, seq_len, hidden_dim), device=device, dtype=dtype, requires_grad=True)
    y = torch.randint(low=0, high=vocab_size, size=(4, seq_len), device=device)

    torch_module = BaselineProjCE(
        hidden_dim, vocab_size
    ).to(device, dtype=dtype)

    triton_module = FusedProjCE(
        hidden_dim, vocab_size
    ).to(device, dtype=dtype)
    
    print(triton_module.w_proj.dtype)

    assert triton_module.w_proj.data.shape == torch_module.w_proj.weight.data.shape
    triton_module.w_proj.data = torch_module.w_proj.weight.data

    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        torch_loss, torch_grad, torch_proj_weight_grad = f(torch_module, x, y)
        triton_loss, triton_grad, triton_proj_weight_grad = f(triton_module, x, y)
    
    assert torch_grad is not None
    assert torch_loss.dtype == triton_loss.dtype, (torch_loss.dtype, triton_loss.dtype)
    assert torch_grad.dtype == triton_grad.dtype, (torch_grad.dtype, triton_grad.dtype)
    assert torch_proj_weight_grad.dtype == triton_proj_weight_grad.dtype, (torch_proj_weight_grad.dtype, triton_proj_weight_grad.dtype)
    
    # These asserts can cause issues when using an autocaster, for correctness the autocaster can be disabled.
    assert torch.allclose(torch_loss, triton_loss, rtol=1e-3), (torch_loss, triton_loss)
    assert torch.allclose(torch_grad, triton_grad, atol=1e-3, rtol=1e-4)
    assert torch.allclose(torch_proj_weight_grad, triton_proj_weight_grad, atol=1e-2, rtol=1e-2)
