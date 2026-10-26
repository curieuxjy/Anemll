#!/usr/bin/env python3
"""Check that --argmax LM-head indices come back exact on ANE (issue #60).

Builds a small split LM head that mirrors the argmax branch in
anemll/ane_converter/*_converter.py, plants a known winner per test case with a
one-hot input, and compares the returned chunk-local index on CPU_ONLY and
CPU_AND_NE. Indices >= 2048 are not all representable in fp16, so any op that
carries the int32 index tensor through the ANE shows up as an off-by-1..4 error.

Patterns:
  squeeze_idx     - 0.3.5: cat(int32 idx) -> squeeze(0).squeeze(0) (the bug)
  squeeze_logits  - fix: squeeze the fp16 logits to [chunk] before argmax

Whether squeeze_idx goes wrong depends on where Core ML places the trailing
squeeze ops, so the CPU_AND_NE compute plan is printed for each pattern. The bug
was reported on M5 Pro / macOS 27.2. On M3 Pro / macOS 27.0.1 those squeezes run
on CPU and both patterns are exact; there, check that squeeze_logits has no op
between reduce_argmax and the argmax_idx output other than cast and concat.

Usage:
  python tests/dev/test_argmax_fp16_index.py
"""
import os
import sys
import tempfile

import numpy as np
import torch
import coremltools as ct

HIDDEN = 1536   # Qwen 2.5 1.5B hidden size
CHUNK = 9496    # 151936 vocab / 16 LM-head chunks
NUM_CHUNKS = 2
LOCAL_TARGETS = (1001, 2049, 3465, 4097, 8195, 9495)
TARGETS = [(c, i) for c in range(NUM_CHUNKS) for i in LOCAL_TARGETS]


class SplitLMHeadArgmax(torch.nn.Module):
    def __init__(self, pattern):
        super().__init__()
        self.pattern = pattern
        self.heads = torch.nn.ModuleList(
            torch.nn.Conv2d(HIDDEN, CHUNK, 1, bias=False) for _ in range(NUM_CHUNKS)
        )

    def forward(self, hidden_states):
        # Same prologue as the converters' LM head wrappers
        hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2)
        logits_list = [h(hidden_states).squeeze(2).transpose(1, 2) for h in self.heads]

        all_idx = []
        all_val = []
        for logits in logits_list:
            if self.pattern == "squeeze_logits":
                logits = logits.squeeze(0).squeeze(0)
            chunk_argmax = torch.argmax(logits, dim=-1, keepdim=True)
            chunk_max_val = torch.gather(logits, -1, chunk_argmax)
            all_idx.append(chunk_argmax.to(torch.int32))
            all_val.append(chunk_max_val)
        if self.pattern == "squeeze_logits":
            return torch.cat(all_idx, dim=-1), torch.cat(all_val, dim=-1)
        argmax_idx = torch.cat(all_idx, dim=-1).squeeze(0).squeeze(0)
        argmax_val = torch.cat(all_val, dim=-1).squeeze(0).squeeze(0)
        return argmax_idx, argmax_val


def build(pattern, out_dir):
    torch.manual_seed(0)
    model = SplitLMHeadArgmax(pattern).eval()
    with torch.no_grad():
        for h in model.heads:
            h.weight.normal_(0, 0.001)
        # Input dim d makes (chunk c, local index i) the winner with logit 1.0
        for d, (c, i) in enumerate(TARGETS):
            model.heads[c].weight[i, d] = 1.0
        sample = torch.zeros(1, 1, HIDDEN)
        traced = torch.jit.trace(model, sample)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="hidden_states", shape=sample.shape, dtype=np.float16)],
        outputs=[
            ct.TensorType(name="argmax_idx", dtype=np.int32),
            ct.TensorType(name="argmax_val", dtype=np.float16),
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )
    path = os.path.join(out_dir, f"{pattern}.mlpackage")
    mlmodel.save(path)
    print_compute_plan(pattern, mlmodel.get_compiled_model_path())
    return {
        cu: ct.models.MLModel(path, compute_units=getattr(ct.ComputeUnit, cu))
        for cu in ("CPU_ONLY", "CPU_AND_NE")
    }


def print_compute_plan(pattern, compiled_path):
    """Print the CPU_AND_NE device placement of every op (the bug depends on it)."""
    from coremltools.models.compute_plan import MLComputePlan

    plan = MLComputePlan.load_from_path(path=compiled_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    print(f"[{pattern}] CPU_AND_NE placement:")
    for fn in plan.model_structure.program.functions.values():
        for op in fn.block.operations:
            if op.operator_name == "const":
                continue
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            device = type(usage.preferred_compute_device).__name__ if usage else "-"
            outputs = ", ".join(o.name for o in op.outputs)
            print(f"  {op.operator_name:<20} {device:<28} -> {outputs}")


def predict(models, compute_unit, d, c):
    x = np.zeros((1, 1, HIDDEN), np.float16)
    x[0, 0, d] = 1.0
    out = models[compute_unit].predict({"hidden_states": x})
    idx = np.ravel(out["argmax_idx"])
    assert idx.shape == (NUM_CHUNKS,), f"unexpected argmax_idx shape {out['argmax_idx'].shape}"
    return int(idx[c])


def main():
    patterns = ("squeeze_idx", "squeeze_logits")
    with tempfile.TemporaryDirectory() as tmp:
        models = {p: build(p, tmp) for p in patterns}

        header = f"{'target':>9}" + "".join(f" | {p + ' ' + cu:>27}" for p in patterns for cu in ("CPU_ONLY", "CPU_AND_NE"))
        print(header)
        wrong = {(p, cu): 0 for p in patterns for cu in ("CPU_ONLY", "CPU_AND_NE")}
        for d, (c, i) in enumerate(TARGETS):
            row = f"c{c} {i:5d}"
            for p in patterns:
                for cu in ("CPU_ONLY", "CPU_AND_NE"):
                    got = predict(models[p], cu, d, c)
                    wrong[(p, cu)] += got != i
                    row += f" | {got:>21d}{'  <-- ' if got != i else '      '}"
            print(row)

    print()
    for (p, cu), n in wrong.items():
        print(f"{p:>15} {cu:>10}: {n}/{len(TARGETS)} wrong")
    return 1 if wrong[("squeeze_logits", "CPU_AND_NE")] else 0


if __name__ == "__main__":
    sys.exit(main())
