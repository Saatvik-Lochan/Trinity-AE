from convert_module import convert_ir_to_triton
import argparse, torch, importlib.util, sys

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--o", type=int, default=0, help="0 only convert, 1 only test, 2 both convert and test")
    parser.add_argument("--m", type=str, default="llama", help="Input model type")
    parser.add_argument("--t", type=str, default="vanilla", help="Benchmark type")
    parser.add_argument("--n", type=int, default=0, help="Case number for IR")
    parser.add_argument("--baseline", nargs="*", default=[], help="List of baselines")
    parser.add_argument("--use_graph", action="store_true")
    parser.add_argument("--print_output", action="store_true")
    args = parser.parse_args()

    num = args.n
    option = args.o
    model = args.m
    target = args.t
    baseline = args.baseline
    use_graph = args.use_graph
    print_output = args.print_output
    dtype = torch.float16

    case_file = f"./evaluation/{target}/{target}_{model}_case{num}.txt"
    output_file = f"./evaluation/{target}/{target}_{model}_benchmark{num}.py"
    module_name = f"{target}_{model}_best"

    output_file = "./evaluation/manual/manual_llama_benchmark1.py"

    if model == 'falcon':
        M = 16
        D = 64
        N = 4544
        P = 1024 - M
        H = 71
        group_size = 4

        constants = {
            'M': M,
            'D': D,
            'N': N,
            'P': P,
            'H': H,
        }
    elif model == 'llama':
        M = 16
        D = 128
        N = 4096
        P = 1024 - M
        H = 32
        num_group = 4

        constants = {
            'M': M,
            'D': D,
            'N': N,
            'P': P,
            'H': H,
        }
    
    tensor_shapes = {
        'X': ('M', 'N'),
        'X2': ('M',),
        'X_norm': ('M', 'N'),

        'WQ': ('N', 'N'),
        'WK': ('N', 'N'),
        'WV': ('N', 'N'),

        'Q1': ('M', 'N'),
        'K1': ('M', 'N'),
        'V1': ('M', 'N'),
        
        'Q2': ('M', 'H', 'D'),
        'K2': ('M', 'H', 'D'),
        'V2': ('M', 'H', 'D'),

        'K_cache': ('H', 'P+M', 'D'),
        'V_cache': ('H', 'P+M', 'D'),

        'Q': ('H', 'M', 'D'),
        'K': ('H', 'M', 'D'),
        'V': ('H', 'M', 'D'),

        'O': ('H', 'M', 'D'),
        'O1': ('M', 'H', 'D'),
        'O2': ('M', 'N'),

        'C': ('H', 'M', 'P+M'),
        'C_exp': ('H', 'M', 'P+M'),
        'C_div': ('H', 'M', 'P+M'),
        'C_sum': ('H', 'M'),
        'noise': ('H', 'M', 'P+M'),
        'C_perturb': ('H', 'M', 'P+M'),
        'C_exp_perturb': ('H', 'M', 'P+M'),
        'C_sum_perturb': ('H', 'M'),
        'C_div_perturb': ('H', 'M', 'P+M'),
        'C_out': ('H', 'P+M'),
        'C_out1': ('H', 'P+M'),
        'C_out2': ('H', 'P+M'),

        'Q_norm': ('H', 'M', 'D'),
        'K_norm': ('H', 'M', 'D'),
    }
        
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Initalize tensors
    std = 0.01
    X = torch.randn((M, N), device=device, dtype=dtype) * std
    X_padded = torch.zeros((16, N), device=device, dtype=dtype) * std
    X_padded[0:M, :] = X

    X2 = torch.zeros((M,), device=device, dtype=dtype) * std
    X_norm = torch.zeros((M, N), device=device, dtype=dtype) * std
    
    WQ = torch.randn((N, N), device=device, dtype=dtype) * std
    WK = torch.randn((N, N), device=device, dtype=dtype) * std
    WV = torch.randn((N, N), device=device, dtype=dtype) * std

    WK_gqa = torch.randn((N, N//num_group), device=device, dtype=dtype) * std
    WV_gqa = torch.randn((N, N//num_group), device=device, dtype=dtype) * std

    Q1 = torch.zeros((M, N), device=device, dtype=dtype) * std
    K1 = torch.zeros((M, N), device=device, dtype=dtype) * std
    V1 = torch.zeros((M, N), device=device, dtype=dtype) * std

    Q2 = torch.zeros((M, H, D), device=device, dtype=dtype) * std
    K2 = torch.zeros((M, H, D), device=device, dtype=dtype) * std
    V2 = torch.zeros((M, H, D), device=device, dtype=dtype) * std
    
    K_cache = torch.randn((H, P+M, D), device=device, dtype=dtype) * std
    V_cache = torch.randn((H, P+M, D), device=device, dtype=dtype) * std

    K_cache_gqa = torch.randn((H//num_group, P+M, D), device=device, dtype=dtype) * std
    V_cache_gqa = torch.randn((H//num_group, P+M, D), device=device, dtype=dtype) * std

    Q = torch.zeros((H, M, D), device=device, dtype=dtype) * std
    K = torch.zeros((H, M, D), device=device, dtype=dtype) * std
    V = torch.zeros((H, M, D), device=device, dtype=dtype) * std

    O = torch.zeros((H, M, D), device=device, dtype=dtype) * std
    O1 = torch.zeros((M, H, D), device=device, dtype=dtype) * std
    O2 = torch.zeros((M, N), device=device, dtype=dtype) * std
    O2 = torch.zeros((16, N), device=device, dtype=dtype) * std

    C = torch.zeros((H, M, P+M), device=device, dtype=dtype) * std
    C_exp = torch.zeros((H, M, P+M), device=device, dtype=dtype) * std
    C_div = torch.zeros((H, M, P+M), device=device, dtype=dtype) * std
    C_sum = torch.zeros((H, M), device=device, dtype=dtype) * std
    
    noise = torch.randn((H, M, P+M), device=device, dtype=dtype) * std
    C_perturb = torch.zeros((H, M, P+M), device=device, dtype=dtype) * std
    C_exp_perturb = torch.zeros((H, M, P+M), device=device, dtype=dtype) * std
    C_div_perturb = torch.zeros((H, M, P+M), device=device, dtype=dtype) * std
    C_sum_perturb = torch.zeros((H, M), device=device, dtype=dtype) * std
    C_out = torch.zeros((H, P+M), device=device, dtype=dtype) * std
    C_out1 = torch.zeros((H, P+M), device=device, dtype=dtype) * std
    C_out2 = torch.zeros((H, P+M), device=device, dtype=dtype) * std

    Q_norm = torch.zeros((H, M, D), device=device, dtype=dtype) * std
    K_norm = torch.zeros((H, M, D), device=device, dtype=dtype) * std


    out = O2.clone()
    ITER = 1000

    match target:
        case "vanilla":
            from baselines import Vanilla, TensorRT_Vanilla, FlashInfer_Vanilla
            trt = TensorRT_Vanilla(M, N, D, H, K_cache.clone(), V_cache.clone(), P, WQ, WK, WV)
            ti = Vanilla(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
            fi = FlashInfer_Vanilla(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
        case "prenorm":
            from baselines import PreNorm, TensorRT_PreNorm, FlashInfer_PreNorm
            trt = TensorRT_PreNorm(M, N, D, H, K_cache.clone(), V_cache.clone(), P, WQ, WK, WV)
            ti = PreNorm(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
            fi = FlashInfer_PreNorm(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
        case "keyformer":
            from baselines import KeyFormer, TensorRT_KeyFormer, FlashInfer_KeyFormer
            trt = TensorRT_KeyFormer(M, N, D, H, K_cache.clone(), V_cache.clone(), P, noise, WQ, WK, WV)
            ti = KeyFormer(M, N, D, P, noise, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
            fi = FlashInfer_KeyFormer(M, N, D, P, noise, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
        case "qknorm":
            from baselines import QKNorm, TensorRT_QKNorm, FlashInfer_QKNorm
            trt = TensorRT_QKNorm(M, N, D, H, K_cache.clone(), V_cache.clone(), P, WQ, WK, WV)
            ti = QKNorm(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
            fi = FlashInfer_QKNorm(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
        case "roco":
            from baselines import RoCo, TensorRT_RoCo, FlashInfer_RoCo
            trt = TensorRT_RoCo(M, N, D, H, K_cache.clone(), V_cache.clone(), P, WQ, WK, WV)
            ti = RoCo(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
            fi = FlashInfer_RoCo(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
        case "gqa":
            from baselines import Vanilla_GQA, TensorRT_Vanilla_GQA
            trt = TensorRT_Vanilla_GQA(M, N, D, H, N//num_group, K_cache.clone(), V_cache.clone(), P, WQ, WK_gqa, WV_gqa)
            ti = Vanilla_GQA(M, N, D, P, N//num_group, K_cache.clone(), V_cache.clone(), WQ, WK_gqa, WV_gqa)

    # --------------- Trinity ---------------------
    print("="*50)
    print(f"Starting Trinity {target}...")
    if len(baseline) == 0 or "trinity" in baseline:
        if option == 0 or option == 2:
            # Convert IR to Triton kernel
            with open(case_file, "r") as f:
                ir = f.read().strip()
            triton_code = convert_ir_to_triton(ir, tensor_shapes, constants)

            with open(output_file, "w") as f:
                f.write(triton_code)
            
            print("="*50)
            print("Triton kernel generated successfully!")
        if option == 0:
            return

        spec = importlib.util.spec_from_file_location(module_name, output_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        forward = getattr(module, "forward")
        # forward = getattr(module, "forward_m1_padded")

        tensor_params = getattr(module, 'TENSOR_PARAMS')
        block_params = getattr(module, 'BLOCK_PARAMS')
        tensors = {
            'X': X, 'X2': X2, 'X_norm': X_norm, 'WQ': WQ, 'WK': WK, 'WV': WV, 'Q1': Q1, 'K1': K1, 'V1': V1,
            'Q2': Q2, 'K2': K2, 'V2': V2, 'K_cache': K_cache, 'V_cache': V_cache, 'Q': Q, 'K': K, 'V': V,
            'O': O, 'O1': O1, 'O2': O2, 'C': C, 'C_exp': C_exp, 'C_div': C_div, 'C_sum': C_sum, 'noise': noise,
            'C_perturb': C_perturb, 'C_exp_perturb': C_exp_perturb, 'C_div_perturb': C_div_perturb, 'C_sum_perturb': C_sum_perturb,
            'C_out': C_out, 'C_out1': C_out1, 'C_out2': C_out2, 'Q_norm': Q_norm, 'K_norm': K_norm,
            'X_padded': X_padded,
        }
        blocks = {
            'block_k': 0, 'block_n': 0, 'block_p': 0
        }
        args = []
        for param in tensor_params:
            if param in tensors:
                args.append(tensors[param])
            else:
                raise ValueError(f"Unknown tensor parameter: {param}")
        for param in block_params:
            if param in blocks:
                args.append(blocks[param])
            else:
                raise ValueError(f"Unknown block parameter: {param}")

        if use_graph:
            stream = torch.cuda.Stream(device)
            with torch.cuda.stream(stream):
                for _ in range(10):
                    forward(*args)
            stream.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream):
                with torch.cuda.graph(graph, stream=stream):
                    forward(*args)
            stream.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                start_event.record()
                for _ in range(ITER):
                    graph.replay()
                end_event.record()
            stream.synchronize()

            time = start_event.elapsed_time(end_event) / ITER
            print(f"Trinity with CUDA Graph: {time} ms")
        else:
            for _ in range(10):
                forward(*args)
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(ITER):
                forward(*args)
            end_event.record()
            torch.cuda.synchronize()

            time = start_event.elapsed_time(end_event) / ITER
            print(f"Trinity without CUDA Graph: {time} ms")
        
        if print_output:
            print(O2)

    # ----------------- TensorRT ---------------------
    if len(baseline) == 0 or "tensorrt" in baseline:
        print("="*50)
        print(f"Starting TensorRT {target}...")

        trt.half()
        if use_graph:
            print(f"TensorRT with CUDA Graph: 0 ms")
        else:
            with torch.no_grad():
                for _ in range(10):
                    out = trt(X)
                torch.cuda.synchronize()

                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for _ in range(ITER):
                    _ = trt(X)
                end_event.record()
                torch.cuda.synchronize()

                time = start_event.elapsed_time(end_event) / ITER
                print(f"TensorRT without CUDA Graph: {time} ms")
        
        if print_output:
            print(out)

    # ----------------- Pytorch Eager ----------------------
    if len(baseline) == 0 or "pytorch" in baseline:
        print("="*50)
        print(f"Starting Pytorch Eager {target}...")
        
        ti = ti.eval()
        if use_graph:
            print(f"Pytorch Eager with CUDA Graph: 0 ms")
        else:
            with torch.no_grad():
                for _ in range(10):
                    out = ti(X)
                torch.cuda.synchronize()

                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for _ in range(ITER):
                    _ = ti(X)
                end_event.record()
                torch.cuda.synchronize()

                time = start_event.elapsed_time(end_event) / ITER
                print(f"Pytorch Eager without CUDA Graph: {time} ms")
        
        if print_output:
            print(out)

    # ----------------- Torch Inductor ---------------------
    if len(baseline) == 0 or "inductor" in baseline:
        print("="*50)
        print(f"Starting Torch Inductor {target}...")

        modes = ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]
        for mode in modes:
            compiled_model = torch.compile(ti, backend="inductor", mode=mode, fullgraph=True)
            for _ in range(10):
                out = compiled_model(X)
            torch.cuda.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(ITER):
                _ = compiled_model(X)
            end_event.record()
            torch.cuda.synchronize()

            time = start_event.elapsed_time(end_event) / ITER
            print(f"Torch Inductor {mode}: {time} ms")
        
        if print_output:
            print(out)
    
    # ----------------- FlashInfer ---------------------
    return
    if len(baseline) == 0 or "flashinfer" in baseline:
        print("="*50)
        print(f"Starting FlashInfer {target}...")
        
        fi.half()
        fi = fi.eval()
        if use_graph:
            print(f"FlashInfer with CUDA Graph: 0 ms")
        else:
            with torch.no_grad():
                for _ in range(10):
                    out = fi(X)
                torch.cuda.synchronize()

                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for _ in range(ITER):
                    _ = fi(X)
                end_event.record()
                torch.cuda.synchronize()

                time = start_event.elapsed_time(end_event) / ITER
                print(f"FlashInfer without CUDA Graph: {time} ms")
        
        if print_output:
            print(out)

if __name__ == "__main__":
    main()