#!/usr/bin/env python3
import argparse
import re
import pandas as pd

def kv_bytes_per_token(layers, kv_heads, head_dim, bytes_per_value):
    # K and V are both stored.
    return 2 * layers * kv_heads * head_dim * bytes_per_value

def parse_model_spec(path):
    text = open(path, encoding="utf-8").read()

    def get_int(pattern):
        m = re.search(pattern, text, flags=re.I)
        if not m:
            raise ValueError(f"Could not find: {pattern}")
        return int(m.group(1))

    return {
        "layers": get_int(r"\|\s*layers\s*\|\s*(\d+)"),
        "kv_heads": get_int(r"\|\s*KV heads \(GQA\)\s*\|\s*(\d+)"),
        "head_dim": get_int(r"\|\s*head_dim\s*\|\s*(\d+)"),
        "max_model_len": get_int(r"\|\s*`max_model_len`\s*\|\s*(\d+)"),
        "gpu_memory_utilization": float(
            re.search(r"\|\s*`gpu_memory_utilization`\s*\|\s*([0-9.]+)", text).group(1)
        ),
        "runtime_overhead_gb": float(
            re.search(r"\|\s*non-KV runtime overhead.*?\|\s*assume\s*~?([0-9.]+)\s*GB", text, flags=re.I).group(1)
        ),
        "params_b": float(
            re.search(r"\|\s*parameters\s*\|\s*([0-9.]+)\s*B", text, flags=re.I).group(1)
        ),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model-spec", required=True)
    args = ap.parse_args()

    spec = parse_model_spec(args.model_spec)
    df = pd.read_csv(args.csv)

    kv = kv_bytes_per_token(
        spec["layers"], spec["kv_heads"], spec["head_dim"], 2
    )
    kv_4096 = kv * spec["max_model_len"]

    # Decimal GB, matching the hardware headline. Weight memory is fp16 = 2 bytes/parameter.
    gpu_budget = 24e9 * spec["gpu_memory_utilization"]
    weights = spec["params_b"] * 1e9 * 2
    kv_budget = gpu_budget - weights - spec["runtime_overhead_gb"] * 1e9
    theoretical_sequences = kv_budget / kv_4096

    print("=== KV CACHE CALCULATION ===")
    print(f"KV bytes/token: {kv:,}")
    print(f"KV bytes/4096-token sequence: {kv_4096:,}")
    print(f"GPU memory budget at utilization limit: {gpu_budget/1e9:.3f} GB")
    print(f"Approx. fp16 weight memory: {weights/1e9:.3f} GB")
    print(f"Non-KV runtime overhead: {spec['runtime_overhead_gb']:.3f} GB")
    print(f"Remaining KV budget: {kv_budget/1e9:.3f} GB")
    print(f"Theoretical full-length sequences: {theoretical_sequences:.2f}")

    df["generated_tok_s"] = df["num_requests"] * df["gen_len"] / df["wall_clock_s"]
    df["input_tok_s"] = df["num_requests"] * df["prompt_len"] / df["wall_clock_s"]
    df["total_tok_s_recomputed"] = (
        df["num_requests"] * (df["prompt_len"] + df["gen_len"])
        / df["wall_clock_s"]
    )
    df["reported_minus_recomputed"] = df["reported_tok_s"] - df["total_tok_s_recomputed"]

    print("\n=== THROUGHPUT RECONCILIATION ===")
    cols = [
        "batch_size", "prompt_len", "gen_len", "wall_clock_s",
        "reported_tok_s", "total_tok_s_recomputed",
        "generated_tok_s", "preempted_seqs", "kv_cache_util"
    ]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    long24 = df[(df.batch_size == 24) & (df.prompt_len == 3584)]
    if not long24.empty:
        r = long24.iloc[0]
        goodput_1 = r.num_requests * r.gen_len / r.wall_clock_s
        goodput_2 = r.reported_tok_s * r.gen_len / (r.prompt_len + r.gen_len)
        print("\n=== BATCH-24 LONG-PROMPT GOODPUT ===")
        print(f"Method 1: 24 * 512 / {r.wall_clock_s:.2f} = {goodput_1:.3f} generated tok/s")
        print(
            f"Method 2: {r.reported_tok_s:.3f} * 512 / "
            f"(3584 + 512) = {goodput_2:.3f} generated tok/s"
        )

    print("\n=== ANOMALY CANDIDATES ===")
    print(df[df["preempted_seqs"] > 0][[
        "batch_size", "prompt_len", "reported_tok_s",
        "ttft_ms_p50", "itl_ms_p50", "e2e_ms_p95",
        "preempted_seqs", "kv_cache_util"
    ]].to_string(index=False))

if __name__ == "__main__":
    main()
