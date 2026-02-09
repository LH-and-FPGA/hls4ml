"""
Convert OpenWakeWord "hey_jarvis_v0.1" model to Vivado HLS using hls4ml.

Usage:
    source ~/.zshrc && conda activate infoproc
    pip install torch onnx   # if not already installed
    python convert_hey_jarvis.py

Target: PYNQ-Z2 (xc7z020clg400-1), Vivado HLS 2020.2

The hey_jarvis model has two sub-models:
  - Primary detector (model.*):   Linear(1536,128) → LN → ReLU → Linear(128,128) → LN → ReLU → Linear(128,1) → Sigmoid
  - Verifier (verifier_model.*):  Linear(1536,64)  → LN → ReLU → Linear(64,64)   → LN → ReLU → Linear(64,1)  → Sigmoid

The original ONNX model uses a conditional If node (run verifier only when primary score >= 0.5).
hls4ml does not support conditional branches, so we convert each sub-model independently.
On FPGA, you can run both and apply the threshold logic in your wrapper.
"""

import sys
import os

# ── Dependency check ─────────────────────────────────────────────────────────
missing = []
for mod in ("torch", "onnx", "onnxruntime", "numpy", "hls4ml"):
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
if missing:
    print(f"Missing packages: {', '.join(missing)}")
    print(f"Install with:  pip install {' '.join(missing)}")
    sys.exit(1)

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

import hls4ml

# ── Configuration ────────────────────────────────────────────────────────────
ONNX_MODEL_PATH = None  # Auto-detected from openwakeword below
OUTPUT_DIR_PRIMARY = "hey_jarvis_primary_hls"
OUTPUT_DIR_VERIFIER = "hey_jarvis_verifier_hls"
FPGA_PART = "xc7z020clg400-1"  # PYNQ-Z2
CLOCK_PERIOD = 10  # ns (100 MHz)
# With 1536 inputs, Dense accumulations can reach ±1536+.
# Need at least 12 integer bits. 32-bit fixed gives good accuracy;
# reduce to ap_fixed<18,12> or ap_fixed<16,10> to save resources after validation.
DEFAULT_PRECISION = "ap_fixed<32,16>"
# PYNQ-Z2 has 220 DSPs and 53K LUTs.
# Linear(1536,128) fully parallel needs 196K multipliers — impossible.
# reuse_factor=256 → ~768 multipliers/cycle (shared via LUTs). Adjust as needed.
DEFAULT_REUSE_FACTOR = 768
STRATEGY = "Resource"  # "Latency" won't fit on PYNQ-Z2 for this model


# ── Locate ONNX model ───────────────────────────────────────────────────────
def find_onnx_model():
    """Find hey_jarvis ONNX model from openwakeword installation."""
    try:
        import openwakeword

        pkg_dir = os.path.dirname(openwakeword.__file__)
        candidate = os.path.join(pkg_dir, "resources", "models", "hey_jarvis_v0.1.onnx")
        if os.path.exists(candidate):
            return candidate
    except ImportError:
        pass

    # Fallback: search common locations
    home = os.path.expanduser("~")
    for root, _, files in os.walk(os.path.join(home, "miniconda3")):
        for f in files:
            if f == "hey_jarvis_v0.1.onnx":
                return os.path.join(root, f)

    print("ERROR: Cannot find hey_jarvis_v0.1.onnx")
    print("Make sure openwakeword is installed and models are downloaded:")
    print("  pip install openwakeword")
    print("  python -c 'from openwakeword.utils import download_models; download_models()'")
    sys.exit(1)


# ── PyTorch model definition ────────────────────────────────────────────────
class WakeWordSubModel(nn.Module):
    """
    Recreates one sub-model (primary or verifier) of hey_jarvis.

    Input shape: (batch, 1536) — flat 2D input.
    We patched hls4ml's PyTorch LayerNorm handler to accept 2D inputs
    (treating them as seq_len=1), which avoids the buggy Dense→Conv1D mapping
    that occurs with 3D inputs.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(1536, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, 1536) — flat 2D input
        x = self.fc1(x)  # (batch, hidden)
        x = self.ln1(x)  # (batch, hidden)
        x = self.relu1(x)
        x = self.fc2(x)  # (batch, hidden)
        x = self.ln2(x)
        x = self.relu2(x)
        x = self.fc3(x)  # (batch, 1)
        x = self.sigmoid(x)
        return x


# ── Weight extraction from ONNX ─────────────────────────────────────────────
def extract_weights(onnx_model):
    """Extract weight tensors from ONNX model initializers."""
    weights = {}
    for init in onnx_model.graph.initializer:
        weights[init.name] = onnx.numpy_helper.to_array(init)
    return weights


def load_weights_into_pytorch(pt_model, onnx_weights, prefix):
    """
    Load ONNX weights into PyTorch model.

    ONNX naming convention from the hey_jarvis model:
      {prefix}.1.weight/bias → fc1
      {prefix}.2.weight/bias → ln1 (gamma/beta)
      {prefix}.4.weight/bias → fc2
      {prefix}.5.weight/bias → ln2 (gamma/beta)
      {prefix}.7.weight/bias → fc3
    """
    mapping = {
        f"{prefix}.1.weight": "fc1.weight",
        f"{prefix}.1.bias": "fc1.bias",
        f"{prefix}.2.weight": "ln1.weight",
        f"{prefix}.2.bias": "ln1.bias",
        f"{prefix}.4.weight": "fc2.weight",
        f"{prefix}.4.bias": "fc2.bias",
        f"{prefix}.5.weight": "ln2.weight",
        f"{prefix}.5.bias": "ln2.bias",
        f"{prefix}.7.weight": "fc3.weight",
        f"{prefix}.7.bias": "fc3.bias",
    }

    state_dict = pt_model.state_dict()
    for onnx_name, pt_name in mapping.items():
        if onnx_name not in onnx_weights:
            print(f"  WARNING: {onnx_name} not found in ONNX model")
            continue
        tensor = torch.from_numpy(onnx_weights[onnx_name].copy())
        if tensor.shape != state_dict[pt_name].shape:
            print(f"  WARNING: shape mismatch for {pt_name}: "
                  f"ONNX={tensor.shape} vs PyTorch={state_dict[pt_name].shape}")
            continue
        state_dict[pt_name] = tensor

    pt_model.load_state_dict(state_dict)
    print(f"  Loaded weights with prefix '{prefix}'")


# ── Verification ─────────────────────────────────────────────────────────────
def verify_pytorch_vs_onnx(pt_model, onnx_path, n_samples=10):
    """Compare PyTorch and ONNX Runtime outputs to ensure weight loading is correct."""
    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name

    pt_model.eval()
    max_diff = 0.0
    for _ in range(n_samples):
        # ONNX expects (1, 16, 96), PyTorch expects (1, 1536)
        x_onnx = np.random.randn(1, 16, 96).astype(np.float32)
        x_pt = torch.from_numpy(x_onnx.reshape(1, 1536))

        onnx_out = sess.run(None, {input_name: x_onnx})[0]
        with torch.no_grad():
            pt_out = pt_model(x_pt).numpy()

        diff = np.abs(onnx_out.flatten() - pt_out.flatten()).max()
        max_diff = max(max_diff, diff)

    return max_diff


# ── hls4ml conversion ────────────────────────────────────────────────────────
def convert_to_hls(pt_model, model_name, output_dir):
    """Convert a PyTorch sub-model to Vivado HLS project."""
    pt_model.eval()
    hidden = pt_model.fc1.out_features  # 128 or 64

    # With 2D input, Dense layers map natively (no Conv1D conversion).
    # ReuseFactor must divide n_in * n_out for Dense Resource strategy.
    # fc2/fc3 have n_in=hidden, so their RF is capped at hidden.
    # fc1 has n_in=1536, so it can use the full DEFAULT_REUSE_FACTOR.
    safe_rf = min(DEFAULT_REUSE_FACTOR, hidden)

    config = hls4ml.utils.config_from_pytorch_model(
        pt_model,
        input_shape=(1536,),  # flat 2D input — avoids Dense→Conv1D mapping
        default_precision=DEFAULT_PRECISION,
        default_reuse_factor=safe_rf,
        granularity="name",
    )
    config["Model"]["Strategy"] = STRATEGY

    # ── Per-layer ReuseFactor for fc1 ──
    # fc1 (1536→hidden) can use a larger RF since n_in=1536 >> hidden.
    fc1_rf = DEFAULT_REUSE_FACTOR  # 1024
    if "fc1" in config.get("LayerName", {}):
        config["LayerName"]["fc1"]["ReuseFactor"] = fc1_rf

    # ── LayerNorm precision ──
    # Wider accum for variance computation accuracy, and wider table for 1/sqrt.
    LAYERNORM_ACCUM = "ap_fixed<48,24>"
    for ln_name in ("ln1", "ln2"):
        if ln_name in config.get("LayerName", {}):
            config["LayerName"][ln_name]["Precision"] = {
                "default": DEFAULT_PRECISION,
                "accum": LAYERNORM_ACCUM,
            }
            config["LayerName"][ln_name]["table_size"] = 8192
            config["LayerName"][ln_name]["table_t"] = LAYERNORM_ACCUM

    print(f"\n  hls4ml config for '{model_name}':")
    print(f"    Precision:    {DEFAULT_PRECISION}")
    print(f"    ReuseFactor:  fc1={fc1_rf}, others={safe_rf}")
    print(f"    Strategy:     {STRATEGY}")
    print(f"    FPGA Part:    {FPGA_PART}")
    print(f"    Clock:        {CLOCK_PERIOD} ns")

    # Convert
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        pt_model,
        input_shape=(1536,),  # flat 2D input
        output_dir=output_dir,
        project_name=model_name,
        backend="Vivado",
        hls_config=config,
        io_type="io_parallel",
        part=FPGA_PART,
        clock_period=CLOCK_PERIOD,
    )

    # Write HLS files to disk (must be before patches)
    hls_model.write()
    print(f"  HLS project written to: {output_dir}/")

    # Patch LayerNorm for wider variance range (table only covers [eps, 1.0])
    patch_layernorm_table_range(output_dir)

    return hls_model


# ── Post-process: patch LayerNorm for C simulation ───────────────────────────
# hls4ml's LayerNorm lookup table only covers variance in [eps, 1.0] (the
# table_range_power2 unsigned config cannot represent negative powers, so
# max_val = 2^(-power) can only be <= 1.0). The hey_jarvis Dense outputs
# have variance ~2-5. We patch the generated HLS to use direct sqrtf for
# accurate C simulation. For HLS synthesis, replace with hls::rsqrt() or
# a custom wider-range table.


PATCHED_LAYERNORM_H = r"""
#ifndef NNET_LAYERNORM_H_
#define NNET_LAYERNORM_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_dense.h"
#include <math.h>

namespace nnet {

struct layernorm_config {
    typedef float bias_t;
    typedef float scale_t;
    typedef float accum_t;
    typedef float table_t;
    static const unsigned n_in = 20;
    static const unsigned seq_len = 4;
    static const unsigned axis = 2;
    static const unsigned epsilon_power_of_10 = 3;
    static const unsigned table_range_power2 = 0;
    static const unsigned table_size = 1024;
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    template <class x_T, class y_T> using product = nnet::product::mult<x_T, y_T>;
};

// Patched layernorm_1d: uses direct 1/sqrtf(var) instead of lookup table.
// The original table only covers variance in [eps, 1.0] which is too narrow
// for models with large feature dimensions. Direct sqrtf is accurate for C
// simulation. For HLS synthesis, replace with hls::rsqrt() or a wider-range table.
template <class data_T, class res_T, typename CONFIG_T>
void layernorm_1d(data_T data[CONFIG_T::n_in / CONFIG_T::seq_len],
                  res_T res[CONFIG_T::n_in / CONFIG_T::seq_len],
                  typename CONFIG_T::scale_t scale[CONFIG_T::n_in / CONFIG_T::seq_len],
                  typename CONFIG_T::bias_t bias[CONFIG_T::n_in / CONFIG_T::seq_len]) {
    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
    #pragma HLS ARRAY_PARTITION variable=data complete
    #pragma HLS ARRAY_PARTITION variable=res complete

    static const unsigned dim = CONFIG_T::n_in / CONFIG_T::seq_len;
    typename CONFIG_T::accum_t sum_cache = 0;
    typename CONFIG_T::accum_t sum_cache2 = 0;
    typename CONFIG_T::accum_t var, mean, diff;
    typename CONFIG_T::accum_t data_diff[dim];
    typename CONFIG_T::accum_t deno_inver;

    #pragma HLS ARRAY_PARTITION variable=data_diff complete

    const typename CONFIG_T::accum_t k_inv = 1.0 / dim;

LAYERNORM_1D_SUM:
    for (int i = 0; i < dim; ++i) {
        sum_cache += static_cast<typename CONFIG_T::accum_t>(data[i]);
    }
    mean = CONFIG_T::template product<typename CONFIG_T::accum_t,
           typename CONFIG_T::accum_t>::product(sum_cache, k_inv);

LAYERNORM_1D_VAR:
    for (int i = 0; i < dim; ++i) {
        data_diff[i] = static_cast<typename CONFIG_T::accum_t>(data[i]) - mean;
        diff = data_diff[i] * data_diff[i];
        sum_cache2 += diff;
    }
    var = CONFIG_T::template product<typename CONFIG_T::accum_t,
          typename CONFIG_T::accum_t>::product(sum_cache2, k_inv);

    // Direct 1/sqrt(var) computation
    {
        float eps_f = powf(10.0f, -(int)CONFIG_T::epsilon_power_of_10);
        float var_f = (float)var;
        if (var_f < eps_f) var_f = eps_f;
        deno_inver = (typename CONFIG_T::accum_t)(1.0f / sqrtf(var_f));
    }

LAYERNORM_1D_RESULT:
    for (int i = 0; i < dim; ++i) {
        res[i] = data_diff[i] * deno_inver * scale[i] + bias[i];
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void layernormalize(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in],
                    typename CONFIG_T::scale_t scale[CONFIG_T::n_in / CONFIG_T::seq_len],
                    typename CONFIG_T::bias_t bias[CONFIG_T::n_in / CONFIG_T::seq_len]) {
    static const unsigned dim = CONFIG_T::n_in / CONFIG_T::seq_len;
    data_T in_val[dim];
    res_T outval[dim];

    #pragma HLS ARRAY_PARTITION variable=scale complete
    #pragma HLS ARRAY_PARTITION variable=bias complete
    #pragma HLS ARRAY_PARTITION variable=in_val complete
    #pragma HLS ARRAY_PARTITION variable=outval complete

LAYERNORM_SEQ_LOOP:
    for (int j = 0; j < CONFIG_T::seq_len; ++j) {
        #pragma HLS PIPELINE
    LAYERNORM_LOAD:
        for (int i = 0; i < dim; ++i) {
            #pragma HLS UNROLL
            in_val[i] = data[j * dim + i];
        }
        layernorm_1d<data_T, res_T, CONFIG_T>(in_val, outval, scale, bias);
    LAYERNORM_STORE:
        for (int i = 0; i < dim; ++i) {
            #pragma HLS UNROLL
            res[j * dim + i] = outval[i];
        }
    }
}

} // namespace nnet
#endif
"""


def patch_layernorm_table_range(output_dir):
    """Replace the generated nnet_layernorm.h with a version using direct sqrtf."""
    ln_path = os.path.join(output_dir, "firmware", "nnet_utils", "nnet_layernorm.h")
    if not os.path.exists(ln_path):
        print(f"  WARNING: {ln_path} not found, skipping LayerNorm patch")
        return
    with open(ln_path, "w") as f:
        f.write(PATCHED_LAYERNORM_H)
    print(f"  Patched LayerNorm: direct sqrtf (C sim accurate) in {ln_path}")


def compile_and_verify(hls_model, pt_model, model_name, n_samples=50):
    """Compile HLS C simulation and compare with PyTorch."""
    print(f"\n  Compiling C simulation for '{model_name}'...")
    # Use _compile() since write() was already called and we patched the files.
    # compile() would call write() again, overwriting our LayerNorm patch.
    hls_model._compile()

    pt_model.eval()
    print(f"  Comparing PyTorch vs HLS (C sim) on {n_samples} samples:")

    np.random.seed(123)
    total_diff = 0.0
    binary_agree = 0
    for i in range(n_samples):
        x_np = np.random.randn(1, 1536).astype(np.float32)
        x_pt = torch.from_numpy(x_np)

        with torch.no_grad():
            pt_out = pt_model(x_pt).numpy().flatten()[0]
        hls_out = hls_model.predict(x_np).flatten()[0]

        diff = abs(pt_out - hls_out)
        total_diff += diff
        agree = (pt_out >= 0.5) == (hls_out >= 0.5)
        binary_agree += int(agree)
        if i < 10:  # Print first 10 samples
            print(f"    Sample {i}: PT={pt_out:.6f}  HLS={hls_out:.6f}  diff={diff:.4f}  {'OK' if agree else 'MISMATCH'}")

    avg_diff = total_diff / n_samples
    print(f"  Mean abs diff: {avg_diff:.4f}")
    print(f"  Binary agreement (threshold=0.5): {binary_agree}/{n_samples} ({100*binary_agree/n_samples:.1f}%)")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # 1. Find model
    global ONNX_MODEL_PATH
    ONNX_MODEL_PATH = find_onnx_model()
    print(f"ONNX model: {ONNX_MODEL_PATH}")

    # 2. Load ONNX and extract weights
    onnx_model = onnx.load(ONNX_MODEL_PATH)
    onnx_weights = extract_weights(onnx_model)
    print(f"Extracted {len(onnx_weights)} weight tensors from ONNX")

    # ── Primary model (hidden=128) ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("PRIMARY MODEL (hidden=128)")
    print("=" * 60)

    primary = WakeWordSubModel(hidden_size=128)
    load_weights_into_pytorch(primary, onnx_weights, prefix="model")

    # Verify against ONNX Runtime (only primary — ONNX output includes If logic)
    # For primary verification, we compare directly
    print("  Verifying PyTorch vs ONNX Runtime...")
    max_diff = verify_pytorch_vs_onnx(primary, ONNX_MODEL_PATH, n_samples=20)
    print(f"  Max abs difference (primary vs ONNX full model): {max_diff:.6f}")
    if max_diff > 0.1:
        print("  NOTE: Larger diffs expected because ONNX model includes verifier branch logic")

    # Convert to HLS
    hls_primary = convert_to_hls(primary, "hey_jarvis_primary", OUTPUT_DIR_PRIMARY)

    # Compile and verify
    compile_and_verify(hls_primary, primary, "primary")

    # ── Verifier model (hidden=64) ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("VERIFIER MODEL (hidden=64)")
    print("=" * 60)

    verifier = WakeWordSubModel(hidden_size=64)
    load_weights_into_pytorch(verifier, onnx_weights, prefix="verifier_model")

    # Convert to HLS
    hls_verifier = convert_to_hls(verifier, "hey_jarvis_verifier", OUTPUT_DIR_VERIFIER)

    # Compile and verify
    compile_and_verify(hls_verifier, verifier, "verifier")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Primary  HLS project: {os.path.abspath(OUTPUT_DIR_PRIMARY)}/")
    print(f"Verifier HLS project: {os.path.abspath(OUTPUT_DIR_VERIFIER)}/")
    print()
    print("Next steps:")
    print("  1. Open the project in Vivado HLS 2020.2")
    print("  2. Run C Synthesis to see resource estimates")
    print("  3. If resources exceed PYNQ-Z2 capacity, increase DEFAULT_REUSE_FACTOR")
    print()
    print("FPGA deployment note:")
    print("  The original model uses conditional logic: run verifier only if primary score >= 0.5")
    print("  On FPGA, you can either:")
    print("    a) Run both models always and apply threshold in wrapper RTL")
    print("    b) Use a state machine to conditionally invoke the verifier IP")


if __name__ == "__main__":
    main()
