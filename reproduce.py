import time
from triton.testing import do_bench
import itertools
import torch
from tqdm import tqdm
import os

from modules import (
    BaselineProjCE,
    FusedProjCE,
)

import matplotlib.pyplot as plt

def save_plots(x_axis, x_axis_name, baseline_results, fused_results, n_chunks_used, filename="plots/plots.png"):
    os.makedirs('plots', exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    
    # Modes to plot
    modes = ['mem', 'time']
    ylabels = ['Memory in (GB)', 'Time (in ms)']
    
    for ax, mode, ylabel in zip(axes, modes, ylabels):
        idx = 0 if mode == 'time' else 1

        num_x = len(x_axis)
        
        # Extract baseline results (since chunking does not affect the baseline we only draw one line)
        group_size_baseline = len(baseline_results) // num_x
        baseline_y = [baseline_results[j * group_size_baseline][idx] for j in range(num_x)]
        
        # Plot baseline
        ax.plot(x_axis, baseline_y, label="Baseline", color="black", marker="o")
        
        # Extract and plot fused results
        num_fused = len(n_chunks_used)
        for i, chunk in enumerate(n_chunks_used):
            fused_y = [fused_results[i + j * num_fused][idx] for j in range(num_x)]
            label = "Dynamic Chunks" if chunk is None else (str(chunk) + ' Chunks')
            ax.plot(x_axis, fused_y, label=label, marker="o")

        ax.set_xlabel(x_axis_name)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('plots/' + filename)
    plt.close()

def baseline_f(m, x, y):
    loss = m(x, y)
    loss.backward()
    return loss

def fused_f(m, x, y):
    loss = m(x.view(-1, x.size(-1)), y.view(-1))
    loss.backward()
    return loss

def benchmark(seq_lens: list[int], hidden_dims: list[int], vocab_sizes: list[int], n_chunks:list, iters: int = 100, warmup: int = 25):
    combined = list(itertools.product(seq_lens, hidden_dims, vocab_sizes, n_chunks))
    
    def run(f, iters, warmup):
        torch.cuda.reset_peak_memory_stats()
        median_runtime = do_bench(f, warmup=warmup, rep=iters)
        peak_mem = torch.cuda.max_memory_allocated()/1e9
        return median_runtime, peak_mem
    
    device = 'cuda'
    dtype = torch.bfloat16
    ctx = torch.amp.autocast(device_type=device, dtype=dtype)
    baseline_results = []
    fused_results = []
    batch = 16
    pbar = tqdm(combined, total=len(combined))
    for combination in pbar:
        seq_len, hidden_dim, vocab_size, n_chunks = combination
        pbar.set_description(f"Benchmarking -> seq_len: {seq_len} | hidden_dim: {hidden_dim} | vocab_size: {vocab_size} | n_chunks: {n_chunks if n_chunks is not None else 'Dynamic'}")
        
        x = torch.randn((batch, seq_len, hidden_dim), dtype=dtype, device=device)
        y = torch.randint(0, vocab_size, size=(batch, seq_len), dtype=torch.long, device=device)
        baseline = BaselineProjCE(hidden_dim, vocab_size)
        fused = FusedProjCE(hidden_dim, vocab_size, n_chunks=n_chunks)
        baseline.to(dtype=dtype, device=device)
        fused.to(dtype=dtype, device=device)
        fs = [lambda: baseline_f(baseline, x, y), lambda: fused_f(fused, x, y)]
        for i, f in enumerate(fs):
            with ctx:
                median_runtime, peak_mem = run(f, iters, warmup)  
            if i == 0:
                baseline_results.append((median_runtime, peak_mem))
            else:
                fused_results.append((median_runtime, peak_mem))
        
    return baseline_results, fused_results
        
def main():
    modes = [2048, 8192, 12576, 25152, 50304]
    n_chunks = [2, 4, 8, None]
    baseline_results, fused_results = benchmark(seq_lens=[1024], hidden_dims=[768], vocab_sizes=modes, n_chunks=n_chunks, warmup=5)
    save_plots(modes, 'Vocab Sizes', baseline_results, fused_results, n_chunks, 'vocab_sizes.png')

if __name__=='__main__':
    main()