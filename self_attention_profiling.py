import torch
import time
import tracemalloc  # For memory profiling
from torch.profiler import profile, ProfilerActivity
import matplotlib.pyplot as plt
import numpy as np

# Define the self-attention mechanism in PyTorch
class SelfAttention(torch.nn.Module):
    def __init__(self, d_model, heads):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.heads = heads
        self.qkv_proj = torch.nn.Linear(d_model, 3 * d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_length, d_model = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_model ** 0.5)
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        output = self.out_proj(attn_output)
        return output

def profile_self_attention(seq_length, d_model=64, heads=8, device='cpu'):
    # Create a random input tensor (batch size = 1 for simplicity)
    x = torch.rand(1, seq_length, d_model).to(device)
    
    # Initialize the self-attention layer
    self_attention = SelfAttention(d_model, heads).to(device)
    
    # Measure memory usage
    tracemalloc.start()
    
    # Measure wall clock time
    start_time = time.time()
    
    # Profile FLOPS and CPU/GPU activity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        _ = self_attention(x)
    
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Extract profiling data (FLOPS and more)
    prof_result = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)
    
    # For simplicity, using cpu_time_total as a rough FLOPS approximation
    flops = sum([item.cpu_time_total for item in prof.key_averages()])
    
    wall_clock_time = end_time - start_time
    memory_usage = peak / (1024 * 1024)  # Convert to MB
    
    return flops, memory_usage, wall_clock_time

# Function to run and collect data for different sequence lengths
def run_experiments(seq_lengths, device='cpu'):
    flops_data, memory_data, time_data = [], [], []
    
    for seq_length in seq_lengths:
        flops, memory, wall_time = profile_self_attention(seq_length, device=device)
        flops_data.append(flops)
        memory_data.append(memory)
        time_data.append(wall_time)
    
    return flops_data, memory_data, time_data

# Define sequence lengths and run the experiment
seq_lengths = [10, 100, 1000, 10000, 100000]

# Run experiments on CPU
cpu_flops, cpu_memory, cpu_time = run_experiments(seq_lengths, device='cpu')

# Plot the results
plt.errorbar(seq_lengths, cpu_flops, yerr=np.std(cpu_flops), label='FLOPS', fmt='-o')
plt.xscale('log')
plt.title('FLOPS vs Sequence Length')
plt.xlabel('Sequence Length')
plt.ylabel('FLOPS (CPU)')
plt.show()

plt.errorbar(seq_lengths, cpu_memory, yerr=np.std(cpu_memory), label='Memory Usage', fmt='-o')
plt.xscale('log')
plt.title('Memory Usage vs Sequence Length')
plt.xlabel('Sequence Length')
plt.ylabel('Memory Usage (MB)')
plt.show()

plt.errorbar(seq_lengths, cpu_time, yerr=np.std(cpu_time), label='Time', fmt='-o')
plt.xscale('log')
plt.title('Wall Clock Time vs Sequence Length')
plt.xlabel('Sequence Length')
plt.ylabel('Wall Clock Time (s)')
plt.show()
