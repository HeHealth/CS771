import torch
import time
import tracemalloc  # For memory profiling on CPU
from torch.profiler import profile, ProfilerActivity
import torch.utils.checkpoint as checkpoint
import matplotlib.pyplot as plt
import numpy as np
from torch.cuda.amp import autocast, GradScaler  # For mixed precision

# Define the self-attention mechanism in PyTorch with checkpointing and chunking
class SelfAttention(torch.nn.Module):
    def __init__(self, d_model, heads, chunk_size=None):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.heads = heads
        self.chunk_size = chunk_size
        self.qkv_proj = torch.nn.Linear(d_model, 3 * d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)

    def forward(self, x):
        return checkpoint.checkpoint(self._attention_forward, x)

    def _attention_forward(self, x):
        batch_size, seq_length, d_model = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        if self.chunk_size is None or seq_length <= self.chunk_size:
            # Process the whole sequence if it's small enough
            scores = torch.matmul(q, k.transpose(-2, -1)) / (d_model ** 0.5)
            attn_weights = torch.nn.functional.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)
        else:
            # Process the sequence in chunks
            attn_output = []
            for i in range(0, seq_length, self.chunk_size):
                q_chunk = q[:, i:i + self.chunk_size, :]
                k_chunk = k[:, i:i + self.chunk_size, :]
                v_chunk = v[:, i:i + self.chunk_size, :]

                scores_chunk = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) / (d_model ** 0.5)
                attn_weights_chunk = torch.nn.functional.softmax(scores_chunk, dim=-1)
                attn_output_chunk = torch.matmul(attn_weights_chunk, v_chunk)
                attn_output.append(attn_output_chunk)

            attn_output = torch.cat(attn_output, dim=1)

        return self.out_proj(attn_output)

# Function to profile the self-attention layer with mixed precision
def profile_self_attention(seq_length, d_model=64, heads=8, device='cpu', chunk_size=None):
    scaler = GradScaler()  # For mixed precision gradient scaling
    x = torch.rand(1, seq_length, d_model).to(device)
    
    # Initialize the self-attention layer with chunking
    self_attention = SelfAttention(d_model, heads, chunk_size).to(device)
    
    if device == 'cpu':
        tracemalloc.start()  # Start memory tracking on CPU

    start_time = time.time()

    # Mixed precision forward pass
    with autocast():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            _ = self_attention(x)

    end_time = time.time()

    if device == 'cpu':
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_usage = peak / (1024 * 1024)  # Convert to MB
    else:
        memory_usage = torch.cuda.max_memory_allocated(device=device) / (1024 * 1024)  # Convert to MB

    flops = sum([item.cpu_time_total for item in prof.key_averages()])
    wall_clock_time = end_time - start_time

    return flops, memory_usage, wall_clock_time

# Function to run experiments on different sequence lengths
def run_experiments(seq_lengths, device='cpu', chunk_size=None):
    flops_data, memory_data, time_data = [], [], []

    for seq_length in seq_lengths:
        flops, memory, wall_time = profile_self_attention(seq_length, device=device, chunk_size=chunk_size)
        flops_data.append(flops)
        memory_data.append(memory)
        time_data.append(wall_time)

    return flops_data, memory_data, time_data

# Define sequence lengths
seq_lengths = [10, 100, 1000, 10000, 100000]

# Run experiments on CPU
cpu_flops, cpu_memory, cpu_time = run_experiments(seq_lengths, device='cpu')

# Run experiments on GPU (with mixed precision and chunking)
chunk_size = 10000  # Adjust the chunk size as needed
if torch.cuda.is_available():
    gpu_flops, gpu_memory, gpu_time = run_experiments(seq_lengths, device='cuda', chunk_size=chunk_size)
else:
    print("CUDA is not available. Skipping GPU tests.")
    gpu_flops, gpu_memory, gpu_time = None, None, None

# Plot the results (FLOPS)
plt.errorbar(seq_lengths, cpu_flops, yerr=np.std(cpu_flops), label='FLOPS (CPU)', fmt='-o')
if gpu_flops is not None:
    plt.errorbar(seq_lengths, gpu_flops, yerr=np.std(gpu_flops), label='FLOPS (GPU)', fmt='-o')
plt.xscale('log')
plt.title('FLOPS vs Sequence Length')
plt.xlabel('Sequence Length')
plt.ylabel('FLOPS')
plt.legend()
plt.savefig('flops_vs_seq_length_comparison_gpu.png')
plt.clf()

# Plot the results (Memory Usage)
plt.errorbar(seq_lengths, cpu_memory, yerr=np.std(cpu_memory), label='Memory Usage (CPU)', fmt='-o')
if gpu_memory is not None:
    plt.errorbar(seq_lengths, gpu_memory, yerr=np.std(gpu_memory), label='Memory Usage (GPU)', fmt='-o')
plt.xscale('log')
plt.title('Memory Usage vs Sequence Length')
plt.xlabel('Sequence Length')
plt.ylabel('Memory Usage (MB)')
plt.legend()
plt.savefig('memory_usage_vs_seq_length_comparison_gpu.png')
plt.clf()

# Plot the results (Wall Clock Time)
plt.errorbar(seq_lengths, cpu_time, yerr=np.std(cpu_time), label='Wall Clock Time (CPU)', fmt='-o')
if gpu_time is not None:
    plt.errorbar(seq_lengths, gpu_time, yerr=np.std(gpu_time), label='Wall Clock Time (GPU)', fmt='-o')
plt.xscale('log')
plt.title('Wall Clock Time vs Sequence Length')
plt.xlabel('Sequence Length')
plt.ylabel('Wall Clock Time (s)')
plt.legend()
plt.savefig('wall_clock_time_vs_seq_length_comparison_gpu.png')
plt.clf()
