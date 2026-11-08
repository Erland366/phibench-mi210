# Speed up a routed block-FP8 Mixture-of-Experts layer at 1024 tokens on AMD MI210

## Objective

The repository at `/app/repo` holds the routed-expert computation of one Mixture-of-Experts (MoE) layer on one
tensor-parallel rank, exactly as a vLLM v0.32 server runs it on **AMD Instinct MI210** (gfx90a, CDNA2, wave64,
104 CUs, BF16 matrix instructions, **no FP8 matrix instructions**). It is correct.

Make `moe_forward` **as fast as possible for a batch of 1024 tokens** without changing its math. Only that
batch size is scored. The score is calibrated on MI210, and the verifier refuses any
other GPU.

## The problem

- 100 routed experts, top-8. Every rank holds all 100 experts, and each MoE call sees every token of the batch.
- Hidden size K = 2560, expert intermediate per rank I = 384: gate/up matmul [768 x 2560], down matmul [2560 x 384].
- Weights: `float8_e4m3fn` with one fp32 scale per 128x128 block. Activations and the output: BF16.
- One call: sort the tokens by expert, gate/up matmul, SiLU-and-mul, down matmul with the router weight, top-k sum.
  The output is this rank's partial sum.

## Editable scope

You may edit **only** these files; any other change scores zero:

```
moe_w8a16.py
configs.json
dec_configs.json
```

You may change the Triton kernels, their launch configuration and the host path in `moe_forward`, and the two tile
tables, including new keys and entries. You may add `precompile(weights, configs) -> None` to `moe_w8a16.py`. The
interfaces of `moe_forward` and `load_configs` must stay the same.

Do not commit: leave your changes in the working tree.

## You must not change

- The math: every top-k slot contributes; the router weight is applied once, on the down matmul output;
  SiLU(gate) x up; FP8 weights read from memory with their block scales (no pre-dequantized, repacked or
  re-quantized copies, no caching of results or weights across calls); BF16 in and out; fp32 accumulation.
- Dependence on the test data: no special-casing of seeds, values, expert ids or routing patterns. The kernel must
  stay correct for any routing and every token count from 1 to 8192, and must stay graph-capturable (no host sync,
  no runtime autotuning).

## Rules the runner enforces

1. **No program waits on another program.** Kernels may not contain a `while` loop, `tl.atomic_cas`, a volatile
   load, a non-relaxed atomic inside a loop, inline assembly with sleeps or branches, or `@triton.autotune`.
2. **Everything compiles before inference.** After `precompile` (or, without one, one call per token count 1-8192),
   no token count under uniform or skewed routing may compile a new Triton variant, and at most 228 variants may
   compile in the warm-up.
3. **Determinism.** Two calls on the same inputs must be bit-identical.

## Correctness

Correctness covers **every** row, not only the scored one. For each row, and for off-grid token counts, against a
float32 reference of the same math: the error (max|out - ref| / max|ref|) must be at most 1.02x the baseline's error
on the same inputs and at most 0.05, and the output must be finite, [M, 2560] and BF16. Any failing case scores zero.
The verifier also uses seeds you do not see.

## Performance

The row `m1024` (1024 tokens) is one HIP graph of whole-layer calls, with the L2 flushed before each replay: the
median of 50 replays, per call. Its routing is drawn from expert counts recorded on a real server. The metric is that
row's time alone.

## Running it

```bash
cd /app/repo
python3 scripts/task_runner.py compile
python3 scripts/task_runner.py correctness
python3 scripts/task_runner.py performance
```

The `performance` command prints every row; read `m1024`. `bash /opt/loop/submit.sh` scores your working tree and
reports your speedup against the baseline. You have up to 16 submissions. `bash /opt/loop/submit.sh --finalize` ends
the session; the best submission is the one graded.
