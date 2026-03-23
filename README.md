# Trinity Project \[ASPLOS 2026\] (To appear)

> Three-Dimensional Tensor Program Optimization via Tile-level Equality Saturation

Trinity is the first tensor program optimizer that achieves scalable joint optimization through tile-level equality saturation. Trinity’s IR can capture the essence of all three optimization axes (algebraic equivalence, memory I/O, compute orchestration). By leveraging equality saturation, Trinity enables scalable joint optimization across the entire graph. 

## Prerequisites

### frontend

```bash
cd frontend
conda env create -f environment.yml
conda activate trinity
cd ..
```

### backend

```bash
cd backend
pip install -r requirements.txt
cd ..
```

### optimizer

```bash
sudo apt install build-essential \
    clang \
    libclang-dev \
    llvm-dev \
    libz3-dev \
    pkg-config
```

## How to Use

You can either run the example script under `scripts/DecAttn.py` or write your own script that defines a PyTorch module and calls `trinity.optimize(...)` with example inputs. Trinity first lowers the module into Trinity IR, then runs optimization over that IR, and finally exports the best Triton kernel under `trinity_output/<basename>/` by default.

For example, you can run an existing example script as:

```bash
python scripts/DecAttn.py
```

> Note: For attention-layer workloads, the optimizer typically spends about 10-15 minutes in equality saturation.

The following example script defines an attention-style PyTorch module, prepares example inputs, and invokes `trinity.optimize(...)` on that module.

File: `scripts/DecAttn.py`

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "frontend"))

import torch
import torch.nn as nn

class Vanilla(nn.Module):
    def __init__(self, M, N, D, P, cache_K, cache_V, W_q=None, W_k=None, W_v=None, device=None, dtype=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D
        self.device = device
        self.dtype = dtype

        self.q_proj = nn.Linear(N, N, bias=False)
        self.k_proj = nn.Linear(N, N, bias=False)
        self.v_proj = nn.Linear(N, N, bias=False)

        self.register_buffer('cache_K', cache_K.to(device))
        self.register_buffer('cache_V', cache_V.to(device))

    def forward(self, X):
        q = self.q_proj(X)
        k = self.k_proj(X)
        v = self.v_proj(X)

        q = q.view(self.M, self.H, self.D)
        k = k.view(self.M, self.H, self.D)
        v = v.view(self.M, self.H, self.D)

        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        self.cache_K[:, self.P:self.P+self.M, :] = k
        self.cache_V[:, self.P:self.P+self.M, :] = v
        cache_K_new = self.cache_K
        cache_V_new = self.cache_V

        q = q.transpose(0, 1)
        scores = torch.matmul(q, cache_K_new.transpose(1, 2))
        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, cache_V_new)
        output = output.transpose(0, 1)
        output = output.contiguous().view(self.M, self.H * self.D)
        return output


if __name__ == "__main__":
    import trinity

    M, N, D, H, P = 16, 4096, 128, 32, 1008

    X = torch.randn((M, N))
    K_cache = torch.randn((H, P + M, D))
    V_cache = torch.randn((H, P + M, D))

    model = Vanilla(M, N, D, P, K_cache, V_cache)
    result = trinity.optimize(model, X, basename="DecAttn", verbose=True)
```

## What `trinity.optimize(...)` Does

`trinity.optimize(...)` takes a PyTorch `nn.Module` and example inputs and runs the following pipeline:

- It first converts the model into a TVM module and lowers it to Relax IR.
- It then lowers the Relax IR into Trinity IR, which is the optimization IR used by Trinity.
- Trinity runs equality saturation on that IR using the `egg` library and uses its cost function to extract candidate kernels.
- Trinity benchmarks the candidate kernels and selects the best-performing kernel.

The generated outputs are written under `trinity_output/<basename>/` by default.

With `basename="roco"`, Trinity creates a workspace under `output_dir/roco/`
(by default `trinity_output/roco/`) and exports files such as:

- `output_dir/roco/config.json`
- `output_dir/roco/frontend/ir.txt`
- `output_dir/roco/frontend/shapes.json`
- `output_dir/roco/frontend/validation_errors.json`
- `output_dir/roco/optimizer/roco_cost6_kern1.txt`
- `output_dir/roco/backend/roco_cost6_kern1_benchmark.json`
- `output_dir/roco/kernels/best_kernel.py`

## Model Requirements

- The input model must be exportable through Trinity's lowering pipeline. In practice, this means:
- `torch.export(...)` must succeed on the model.
- TVM Relax must be able to import the resulting `ExportedProgram`.
- The lowered IR must use operations currently supported by Trinity. For example, `torch.topk` and `torch.split` are currently not supported.
