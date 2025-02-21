import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import triton
import triton.language as tl

MAX_FUSED_SIZE = 65536 // 2

class BaselineProjCE(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.w_proj = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(self, x, y):
        logits = self.w_proj(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        return loss

@triton.jit
def fused_proj_ce_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    loss_ptr,
    loss_stride,
    n_cols,
    n_elements,
    ignore_index,
    BLOCK_SIZE: tl.constexpr,
):  
    # https://github.com/triton-lang/triton/issues/1058
    # cast to avoid overflow
    row_idx = tl.program_id(0).to(tl.int64)
    
    Y_ptr += row_idx * Y_stride
    X_ptr += row_idx * X_stride
    loss_ptr += row_idx * loss_stride
    
    y = tl.load(Y_ptr)
    
    if y == ignore_index:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return
    
    # Online softmax: https://arxiv.org/pdf/1805.02867 (algo 3)
    m = float('-inf')
    d = 0.0
    ori_X_y = tl.load(
        X_ptr + y
    )  # we need to store the original value of X_y for the loss calculation
    for j in range(0, n_cols, BLOCK_SIZE):
        X_offsets = j + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets, mask=X_offsets < n_cols, other=float("-inf")
        )
        block_max = tl.max(X_block)
        next_m = tl.maximum(m, block_max)
        d = d * tl.exp(m - next_m) + tl.sum(tl.exp(X_block - next_m))
        m = next_m
        
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets, mask=X_offsets < n_cols, other=float("-inf")
        )
        X_block = (tl.exp(X_block - m) / d) / (n_elements)
        tl.store(X_ptr + X_offsets, X_block, mask=X_offsets < n_cols)
        
    # We need tl.debug_barrier() to ensure the new result of X_ptr is written as mentioned in
    # https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/ops/cross_entropy.py#L34
    tl.debug_barrier()
    
    loss = -(ori_X_y - m - tl.log(d))
    
    X_y = tl.load(X_ptr + y)
    X_y += -1 / (n_elements)
    
    tl.store(loss_ptr, loss)
    tl.store(X_ptr + y, X_y)

class FusedAutoGradFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, w_proj, n_chunks=None, ignore_index=-100):
        seq_len, hidden_dim = x.shape
        device = x.device
        dtype = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled() else x.dtype
        
        if dtype != x.dtype:
            # cast x to the autocasted dtype used
            x = x.to(dtype=dtype)
        if dtype != w_proj.dtype:
            # cast the weights to the autocasted dtype used
            w_proj = w_proj.to(dtype=dtype)
            
        # loss can be full precision
        loss = torch.empty(seq_len, dtype=torch.float32, device=device)
        in_grad = torch.zeros_like(x)
        w_proj_grad = torch.zeros_like(w_proj)
        vocab_size = w_proj.size(0)
        
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(vocab_size))

        if n_chunks is None:
            inc_factor = triton.cdiv(vocab_size, hidden_dim)
            CHUNK_SIZE = triton.next_power_of_2(
                triton.cdiv(seq_len, inc_factor)
            )
            CHUNKS = triton.cdiv(seq_len, CHUNK_SIZE)
        else:
            CHUNK_SIZE = seq_len // n_chunks
            CHUNKS = n_chunks
        
        n_elements = (y != ignore_index).sum().item()
        for i, chunk in enumerate(torch.split(x, CHUNK_SIZE)):
            chunk_start = i * CHUNK_SIZE
            chunk_end = min((i + 1) * CHUNK_SIZE, seq_len)
            
            chunk_logits = chunk @ w_proj.T
            chunk_y = y[chunk_start:chunk_end]
            chunk_loss = loss[chunk_start:chunk_end]
            chunk_in_grad = in_grad[chunk_start:chunk_end]
            chunk_w_proj_grad = w_proj_grad[chunk_start:chunk_end]
            
            # always do softmax at full precision
            chunk_logits = chunk_logits.to(torch.float32)
            
            # make sure the tensors are contiguous
            chunk_logits = chunk_logits.contiguous()
            chunk_y = chunk_y.contiguous()
            
            chunk_logits_grad = chunk_logits
            chunk_n_elements = (chunk_y != ignore_index).sum().item()

            # we use chunk_end - chunk_start to account for cases
            # where the input is not divisable by the number of chunks.
            fused_proj_ce_kernel[(chunk_end - chunk_start,)](
                chunk_logits,
                chunk_logits.stride(-2),
                chunk_y,
                chunk_y.stride(-1),
                chunk_loss,
                chunk_loss.stride(-1),
                vocab_size,
                chunk_n_elements,
                ignore_index,
                num_warps=32,
                BLOCK_SIZE=CHUNK_SIZE
            )
            # cast back to the autocasted (or not) dtype
            chunk_logits = chunk_logits.to(dtype)
            
            # scale the gradient by the ratio of ignored indices
            chunk_logits_grad = chunk_logits * (
                chunk_n_elements / n_elements
            )
            in_grad[chunk_start:chunk_end] = chunk_logits_grad @ w_proj
            torch.addmm(
                input=w_proj_grad,
                mat1=chunk_logits.t(),
                mat2=chunk,
                out=w_proj_grad,
                alpha=chunk_n_elements / n_elements,
            )

        cum_loss = torch.sum(loss) / n_elements
        ctx.save_for_backward(in_grad.detach(), w_proj_grad.detach())
        return cum_loss
    
    @staticmethod
    def backward(ctx, grad_output):
        in_grad, w_proj_grad = ctx.saved_tensors
        return (in_grad * grad_output, None, w_proj_grad * grad_output, None, None)

class FusedProjCE(nn.Module):
    def __init__(self, hidden_size, vocab_size, n_chunks=None):
        super().__init__()
        self.w_proj = nn.Parameter(torch.empty(vocab_size, hidden_size))
        nn.init.kaiming_uniform_(self.w_proj, a=math.sqrt(2))
        self.n_chunks = n_chunks
        
    def forward(self, x, y):
        loss = FusedAutoGradFunc.apply(x, y, self.w_proj, self.n_chunks)
        return loss
