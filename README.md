# Trinity Project \[ASPLOS 2026\]

> Three-Dimensional Tensor Program Optimization via Tile-level Equality Saturation

Trinity is the first tensor program optimizer that achieves scalable joint optimization through tile-level equality saturation. Trinity's IR can capture the essence of all three optimization axes (algebraic equivalence, memory I/O, compute orchestration). By leveraging equality saturation, Trinity enables scalable joint optimization across the entire graph.

## Project Structure

```
Trinity-AE/
├── optimizer/      # Tile-level equality saturation (Rust)
├── backend/        # IR → Triton kernel generation & profiling
```

For detailed documentation, see the README in each directory:
- [`optimizer/`](optimizer/README.md) — Equality saturation engine
- [`backend/`](backend/README.md) — Triton code generation and profiling
