import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import importlib.util
import json
import tempfile
import time
import traceback
from typing import Dict, List, Optional

import torch

from ffn_llama_benchmark import (
    BenchmarkResult,
    LlamaBenchmark,
    print_comprehensive_report,
    save_incremental_results,
    save_top_k_results,
)


class BlockbusterBenchmark(LlamaBenchmark):
    def __init__(self, tensor_config: Dict[str, int], device):
        self.M = tensor_config["M"]
        self.D = tensor_config["D"]
        self.K = tensor_config["K"]
        self.N = tensor_config["N"]

        self.tensor_config = tensor_config

        self.tensor_shapes = {
            "X": (self.M, self.D),
            "X_rowsum": (self.M,),
            "X_norm": (self.M, self.D),
            "W": (self.K, self.D),
            "V": (self.K, self.D),
            "FF1a": (self.M, self.K),
            "FF1b": (self.M, self.K),
            "FF1a_silu": (self.M, self.K),
            "FF1": (self.M, self.K),
            "O": (self.M, self.N),
            "U": (self.N, self.K),
        }

        self.shape_dict = {
            "X": ("M", "D"),
            "X_rowsum": ("M",),
            "X_norm": ("M", "D"),
            "W": ("K", "D"),
            "V": ("K", "D"),
            "FF1a": ("M", "K"),
            "FF1b": ("M", "K"),
            "FF1a_silu": ("M", "K"),
            "FF1": ("M", "K"),
            "O": ("M", "N"),
            "U": ("N", "K"),
        }

        self.const_dict = tensor_config.copy()

        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(self.device)
        print(f"GPU: {torch.cuda.get_device_name(self.device)}")

        self.create_test_tensors()
        self._temp_files = []

    def create_test_tensors(self):
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        self.tensors = {}
        for name, shape in self.tensor_shapes.items():
            if name in ["X_rowsum", "X_norm", "FF1a", "FF1b", "FF1a_silu", "FF1", "O"]:
                self.tensors[name] = torch.zeros(shape, dtype=torch.float16, device=self.device)
            else:
                self.tensors[name] = torch.randn(shape, dtype=torch.float16, device=self.device).clamp(-1, 1) * 0.001

    def compile_and_load_kernel(self, kernel_code: str, kernel_id: int):
        temp_file = None
        try:
            module_name = f"blockbuster_kernel_{kernel_id}"
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(kernel_code)
                temp_file = f.name

            spec = importlib.util.spec_from_file_location(module_name, temp_file)
            if spec is None or spec.loader is None:
                print("Error: Failed to create import spec for generated kernel")
                return None
            module = importlib.util.module_from_spec(spec)
            module.__file__ = temp_file

            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            self._temp_files = getattr(self, "_temp_files", [])
            self._temp_files.append(temp_file)

            if hasattr(module, "forward"):
                return module
            print("Error: Cannot call the kernel")
            return None

        except Exception as e:
            print(f"Error compiling kernel: {e}")
            traceback.print_exc()
            if temp_file is not None and os.path.exists(temp_file):
                with open(temp_file, "r") as f:
                    print("Generated kernel code:")
                    print(f.read())
                os.unlink(temp_file)
            return None

    def benchmark_kernel(self, kernel_module, warmup_runs: int = 10, benchmark_runs: int = 100) -> float:
        try:
            tensor_params = getattr(
                kernel_module,
                "TENSOR_PARAMS",
                ["X", "X_rowsum", "X_norm", "W", "V", "FF1a", "FF1b", "FF1a_silu", "FF1", "O", "U"],
            )
            kernel_fn = kernel_module.forward

            for name in ["X_rowsum", "X_norm", "FF1a", "FF1b", "FF1a_silu", "FF1", "O"]:
                if name in self.tensors:
                    self.tensors[name].zero_()

            args = []
            for param in tensor_params:
                if param in self.tensors:
                    args.append(self.tensors[param])
                elif param in self.tensor_shapes:
                    args.append(torch.zeros(self.tensor_shapes[param], dtype=torch.float16, device=self.device))
                else:
                    raise ValueError(f"Unknown tensor parameter: {param}")

            kernel_fn(*args)
            torch.cuda.synchronize()

            for _ in range(warmup_runs):
                kernel_fn(*args)
            torch.cuda.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            for _ in range(benchmark_runs):
                kernel_fn(*args)
            end_event.record()
            torch.cuda.synchronize()

            return start_event.elapsed_time(end_event) / benchmark_runs

        except Exception as e:
            print(f"Error benchmarking kernel: {e}")
            traceback.print_exc()
            raise

    def run_single_benchmark(self, ir_id: int, ir_expr: str) -> BenchmarkResult:
        try:
            kernel_code = self.generate_kernel_code(ir_expr)
            if kernel_code is None:
                return BenchmarkResult(ir_id, ir_expr, float("inf"), self.tensor_config, "Failed to generate kernel")

            kernel_module = self.compile_and_load_kernel(kernel_code, ir_id)
            if kernel_module is None:
                return BenchmarkResult(ir_id, ir_expr, float("inf"), self.tensor_config, "Failed to compile kernel")

            exec_time = self.benchmark_kernel(kernel_module)
            self.cleanup_gpu()

            module_name = f"blockbuster_kernel_{ir_id}"
            if module_name in sys.modules:
                del sys.modules[module_name]

            return BenchmarkResult(ir_id, ir_expr, exec_time, self.tensor_config)

        except Exception as e:
            self.cleanup_gpu()
            return BenchmarkResult(ir_id, ir_expr, float("inf"), self.tensor_config, str(e))


def run_comprehensive_benchmark(tensor_configs, ir_file, start_expressions, num_expressions, top_k, output_file, device):
    all_results = []
    benchmark_instances = []

    print("Running comprehensive benchmark with:")
    print(f"  - {len(tensor_configs)} tensor configurations")
    print()

    with open(output_file, "w") as f:
        json.dump([], f)

    for tensor_idx, tensor_config in enumerate(tensor_configs):
        print(
            f"\nTensor Configuration {tensor_idx + 1}/{len(tensor_configs)}: "
            f"M={tensor_config['M']}, D={tensor_config['D']}, K={tensor_config['K']}, N={tensor_config['N']}"
        )

        benchmark = BlockbusterBenchmark(tensor_config, device)
        benchmark_instances.append(benchmark)

        try:
            results = benchmark.run_all_benchmarks(ir_file, min_expressions=start_expressions, num=num_expressions)
            config_results = {
                "tensor_config": tensor_config,
                "results": results,
            }
            all_results.append(config_results)
            save_incremental_results(config_results, output_file)
            print(f"  Saved results for configuration {tensor_idx + 1}/{len(tensor_configs)}")

        except Exception as e:
            print(f"  Error in configuration: {str(e)}")
            error_result = {
                "tensor_config": tensor_config,
                "error": str(e),
                "results": [],
            }
            all_results.append(error_result)
            save_incremental_results(error_result, output_file)

        benchmark.cleanup()

    return all_results, benchmark_instances


def main():
    IR_FILE = "./evaluation/ffn/blockbuster_rms_ffn_swiglu_cost6_kern5_wo_scheduler2.txt"
    OUTPUT_FILE = "./evaluation/ffn/blockbuster_rms_ffn_swiglu.json"
    START_EXPRESSIONS = 0
    NUM_EXPRESSIONS = 10
    TOP_K = 5
    TENSOR_CONFIGS = [{"M": 12800, "D": 576, "K": 1536, "N": 576}]

    parser = argparse.ArgumentParser(description="Run comprehensive Attacc IR benchmarks")
    parser.add_argument("--ir", type=str, default=IR_FILE, help="Path to the IR expressions file")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE, help="Path to save benchmark results")
    parser.add_argument("--start", type=int, default=START_EXPRESSIONS, help="Start from test case ID")
    parser.add_argument("--device", type=int, default=0, help="CUDA device number")
    parser.add_argument("--num", type=int, default=NUM_EXPRESSIONS, help="Number of expressions to benchmark")
    parser.add_argument("--end", action="store_true", help="Run from start ID to the last test case")
    parser.add_argument("--topk", type=int, default=TOP_K, help="Number of top kernels to report")
    parser.add_argument("--all", action="store_true", help="Run all configurations comprehensively")

    args = parser.parse_args()

    if args.end and args.num != NUM_EXPRESSIONS:
        print("Error: Cannot use --end and --num together. Use either --num or --end, not both.")
        return

    if args.all:
        with open(args.ir, "r") as f:
            total_expressions = len(f.readlines())
    elif args.end:
        total_expressions = None
    else:
        total_expressions = args.num

    if not torch.cuda.is_available():
        print("Error: CUDA device not available. Triton requires CUDA.")
        return

    print("Starting comprehensive Attacc benchmarks...")
    all_results, _ = run_comprehensive_benchmark(
        TENSOR_CONFIGS,
        args.ir,
        args.start,
        total_expressions,
        args.topk,
        args.output,
        args.device,
    )

    print(f"\nAll results saved to: {args.output}")
    final_result = print_comprehensive_report(all_results, args.topk)
    final_output = args.output.replace(".json", f"_top{args.topk}.json")
    save_top_k_results(final_result, final_output)


if __name__ == "__main__":
    main()
