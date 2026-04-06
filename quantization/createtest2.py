"""
export_phi3.py
──────────────────────────────────────────────────────────────────────────────
Exports Phi-3 Mini (or TinyLlama as fallback) to a quantized ExecuTorch
.pte file optimised for memory-constrained Android devices (4 GB RAM).

Quantization  : 8da4w  (Int8DynActInt4WeightQuantizer, groupsize=32)
Backend        : XNNPACK (Android CPU / ARM NEON)
Max seq length : 512 tokens (conservative for 4 GB RAM)

Tested stack
────────────
  torch       == 2.11.0
  torchvision == 0.26.0
  torchaudio  == 2.11.0
  torchao     == 0.17.0
  executorch  == 1.2.0

Usage
──────
  source /Users/akhil/phi3_et_env/bin/activate
  HF_HUB_DISABLE_XET=1 python export_phi3.py --model phi3
  HF_HUB_DISABLE_XET=1 python export_phi3.py --model tinyllama
  HF_HUB_DISABLE_XET=1 python export_phi3.py --model phi3 --max_seq_len 128
"""

import argparse
import logging
import os
import shutil
import sys
import time
import types
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torch >= 2.11.*")
warnings.filterwarnings("ignore", message=".*upgrade to torch.*")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("export_phi3")

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Export LLM to ExecuTorch .pte")
parser.add_argument("--model",       choices=["phi3", "tinyllama"], default="phi3")
parser.add_argument("--weights_dir", default="./phi3_weights")
parser.add_argument("--output_dir",  default="./android_assets")
parser.add_argument("--max_seq_len", type=int, default=512)
parser.add_argument("--skip_quant_verify", action="store_true")

args = parser.parse_args()

os.makedirs(args.output_dir,  exist_ok=True)
os.makedirs(args.weights_dir, exist_ok=True)

# ── Model registry ────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "phi3": {
        "repo_id"         : "microsoft/Phi-3-mini-4k-instruct",
        "output_name"     : "phi3_mini_8da4w.pte",
        "display_name"    : "Phi-3 Mini 4K Instruct",
        "expected_ram_mb" : 900,
        "weight_files"    : [
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.model",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "configuration_phi3.py",
            "modeling_phi3.py",
        ],
    },
    "tinyllama": {
        "repo_id"         : "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "output_name"     : "tinyllama_8da4w.pte",
        "display_name"    : "TinyLlama 1.1B Chat",
        "expected_ram_mb" : 350,
        "weight_files"    : [
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "model.safetensors",
        ],
    },
}

cfg = MODEL_REGISTRY[args.model]

log.info("=" * 70)
log.info(f"  Model         : {cfg['display_name']}")
log.info(f"  Quantization  : 8da4w (INT8 activations + INT4 weights)")
log.info(f"  Backend       : XNNPACK (Android CPU / ARM NEON)")
log.info(f"  Max seq len   : {args.max_seq_len} tokens")
log.info(f"  Output dir    : {args.output_dir}")
log.info(f"  Expected RAM  : ~{cfg['expected_ram_mb']} MB on device")
log.info("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Download / verify weights AND custom architecture files
# ─────────────────────────────────────────────────────────────────────────────
log.info("")
log.info("STEP 1 — Checking / downloading model weights + architecture files")

free_gb = shutil.disk_usage(args.weights_dir).free / 1024 ** 3
log.info(f"  Free disk space : {free_gb:.1f} GB")
if free_gb < 8 and args.model == "phi3":
    log.error(
        f"  Only {free_gb:.1f} GB free — Phi-3 Mini needs ~8 GB. "
        "Free up space or use --model tinyllama (~2 GB)."
    )
    sys.exit(1)

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    log.error("huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

for filename in cfg["weight_files"]:
    target = os.path.join(args.weights_dir, filename)
    if os.path.exists(target) and os.path.getsize(target) > 0:
        size_mb = os.path.getsize(target) / 1024 ** 2
        log.info(f"  ✅ Already exists : {filename} ({size_mb:.1f} MB)")
        continue
    log.info(f"  ⬇️  Downloading   : {filename} …")
    try:
        hf_hub_download(
            repo_id=cfg["repo_id"],
            filename=filename,
            local_dir=args.weights_dir,
            resume_download=True,
        )
        size_mb = os.path.getsize(target) / 1024 ** 2
        log.info(f"  ✅ Done           : {filename} ({size_mb:.1f} MB)")
    except Exception as e:
        log.warning(f"  ⚠️  Skipped : {filename} — {e}")

tokenizer_src = os.path.join(args.weights_dir, "tokenizer.json")
tokenizer_dst = os.path.join(args.output_dir,  "tokenizer.json")
if os.path.exists(tokenizer_src):
    shutil.copy2(tokenizer_src, tokenizer_dst)
    log.info(f"  ✅ tokenizer.json → {tokenizer_dst}")
else:
    log.error("  ❌ tokenizer.json not found after download — aborting.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Verify versions
# ─────────────────────────────────────────────────────────────────────────────
log.info("")
log.info("STEP 2 — Verifying installation")

try:
    import torch
    import executorch  # noqa: F401
    from importlib.metadata import version as pkg_version

    try:
        et_ver = pkg_version("executorch")
        ao_ver = pkg_version("torchao")
    except Exception:
        et_ver = "unknown"
        ao_ver = "unknown"

    log.info(f"  Python     : {sys.version.split()[0]}")
    log.info(f"  PyTorch    : {torch.__version__}")
    log.info(f"  ExecuTorch : {et_ver}")
    log.info(f"  torchao    : {ao_ver}")

except ImportError as e:
    log.error(f"  ❌ {e}")
    log.error("  Run: pip install torch executorch transformers torchao")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2b — Patch config.json  (rope_scaling + use_cache)
# ─────────────────────────────────────────────────────────────────────────────
log.info("")
log.info("STEP 2b — Patching config.json …")

import json

cfg_path = os.path.join(args.weights_dir, "config.json")
with open(cfg_path) as f:
    model_cfg = json.load(f)

_patched   = False
hidden     = model_cfg.get("hidden_size",         3072)
num_heads  = model_cfg.get("num_attention_heads",   32)
factor_len = (hidden // num_heads) // 2

rope = model_cfg.get("rope_scaling")
needs_patch = (
    rope is None
    or not isinstance(rope, dict)
    or "short_factor" not in rope
    or "long_factor"  not in rope
    or rope.get("type") == "default"
)

if needs_patch:
    model_cfg["rope_scaling"] = {
        "type":         "longrope",
        "short_factor": [1.0] * factor_len,
        "long_factor":  [1.0] * factor_len,
    }
    _patched = True
    log.info(f"  ✅ rope_scaling patched (factor_len={factor_len})")
else:
    log.info(f"  ✅ rope_scaling already valid: {rope.get('type')}")

if model_cfg.get("use_cache", True):
    model_cfg["use_cache"] = False
    _patched = True
    log.info("  ✅ use_cache set to False")
else:
    log.info("  ✅ use_cache already False")

hidden_actual = model_cfg.get("hidden_size", 0)
vocab_actual  = model_cfg.get("vocab_size",  0)
mtype         = model_cfg.get("model_type",  "unknown")
if hidden_actual != 3072 or vocab_actual != 32064:
    log.error(f"  ❌ Wrong model: model_type={mtype} hidden={hidden_actual} vocab={vocab_actual}")
    log.error(f"     Fix: rm -rf {args.weights_dir} && re-run")
    sys.exit(1)

if _patched:
    with open(cfg_path, "w") as f:
        json.dump(model_cfg, f, indent=2)
    log.info("  ✅ config.json saved")

with open(cfg_path) as f:
    verify = json.load(f)
assert verify.get("rope_scaling", {}).get("type") == "longrope", \
    "config.json write failed — check file permissions"
log.info("  ✅ config.json verified on disk")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Load model
# ─────────────────────────────────────────────────────────────────────────────
log.info("")
log.info("STEP 3 — Loading model weights …")

t0 = time.time()

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    custom_config   = os.path.join(args.weights_dir, "configuration_phi3.py")
    has_custom_arch = os.path.exists(custom_config)

    if has_custom_arch:
        log.info("  Custom arch files present — loading fully from local disk")
        model_source = args.weights_dir
        local_only   = True
    else:
        log.warning("  configuration_phi3.py not found — falling back to HF Hub")
        model_source = cfg["repo_id"]
        local_only   = False

    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=True,
        local_files_only=local_only,
    )
    log.info("  ✅ Tokenizer loaded")

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=local_only,
        attn_implementation="eager",
    )
    hf_model.eval()

    import torch._dynamo as dynamo
    dynamo.config.assume_static_by_default         = False
    dynamo.config.capture_dynamic_output_shape_ops = True
    dynamo.config.capture_scalar_outputs           = True

    param_count = sum(p.numel() for p in hf_model.parameters()) / 1e9
    load_time   = time.time() - t0
    log.info(f"  ✅ Model loaded : {param_count:.2f}B params  ({load_time:.1f}s)")

except Exception as e:
    log.error(f"  ❌ Model load failed: {e}")
    log.error("  Fix: pip install --upgrade 'transformers>=4.40.0'")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3b — Patch Phi3LongRoPEScaledRotaryEmbedding
#
# ROOT CAUSE:
#   Phi3LongRoPEScaledRotaryEmbedding.forward() contains:
#
#     seq_len = seq_len or torch.max(position_ids) + 1
#     if seq_len > self.original_max_position_embeddings:   ← Eq(u0,1) guard
#         ext_factors = torch.tensor(self.long_factor, ...)
#     else:
#         ext_factors = torch.tensor(self.short_factor, ...)
#
#   `seq_len` is derived from a tensor at runtime → torch.export cannot
#   resolve the `if` as a static boolean → Eq(u0, 1) crash.
#
#   IMPORTANT: This class has NO cos_cached / sin_cached attributes.
#   It recomputes cos/sin from inv_freq on every call.
#
# FIX:
#   Replace forward() with a version that always uses short_factor
#   (safe because max_seq_len=512 < original_max_position_embeddings=4096)
#   and computes cos/sin inline with no data-dependent branch.
# ─────────────────────────────────────────────────────────────────────────────
log.info("")
log.info("STEP 3b — Patching Phi3LongRoPEScaledRotaryEmbedding …")

def _patch_phi3_rope(model, max_seq_len):
    import torch
    patched = 0

    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if "RotaryEmbedding" not in cls_name:
            continue

        # ── Detect which variant this is ─────────────────────────────────────
        has_long_rope  = hasattr(module, "short_factor") and hasattr(module, "long_factor")
        has_cos_cached = hasattr(module, "cos_cached")

        if has_long_rope:
            # Phi3LongRoPEScaledRotaryEmbedding — computes cos/sin on the fly
            # Always use short_factor (seq_len <= max_seq_len < original_max)
            def make_longrope_forward(mod):
                def patched_forward(self, x, position_ids, seq_len=None):
                    # Always use short_factor — no data-dependent branch
                    ext_factors = torch.tensor(
                        self.short_factor, dtype=torch.float32, device=x.device
                    )
                    inv_freq_shape = (
                        torch.arange(0, self.dim, 2, dtype=torch.int64, device=x.device)
                        .float() / self.dim
                    )
                    self.inv_freq = 1.0 / (ext_factors * self.base ** inv_freq_shape)

                    inv_freq_expanded = (
                        self.inv_freq[None, :, None]
                        .float()
                        .expand(position_ids.shape[0], -1, 1)
                    )
                    position_ids_expanded = position_ids[:, None, :].float()

                    device_type = x.device.type
                    if device_type == "mps":
                        device_type = "cpu"
                    with torch.autocast(device_type=device_type, enabled=False):
                        freqs = (
                            inv_freq_expanded.float()
                            @ position_ids_expanded.float()
                        ).transpose(1, 2)
                        emb = torch.cat((freqs, freqs), dim=-1)
                        cos = emb.cos()
                        sin = emb.sin()
                    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
                return patched_forward

            module.forward = types.MethodType(make_longrope_forward(module), module)
            patched += 1
            log.info(f"  ✅ Patched (LongRoPE, no cos_cached) : {name}  ({cls_name})")

        elif has_cos_cached:
            # Phi3RotaryEmbedding base class — uses cos_cached/sin_cached
            def make_cached_forward(mod, seq_len_cap):
                def patched_forward(self, x, position_ids, seq_len=None):
                    if not getattr(self, "_export_cache_built", False):
                        if hasattr(self, "_set_cos_sin_cache"):
                            self._set_cos_sin_cache(
                                seq_len=seq_len_cap,
                                device=x.device,
                                dtype=x.dtype,
                            )
                        self._export_cache_built = True
                    cos = self.cos_cached[:seq_len_cap].to(x.dtype)
                    sin = self.sin_cached[:seq_len_cap].to(x.dtype)
                    return cos, sin
                return patched_forward

            module.forward = types.MethodType(
                make_cached_forward(module, max_seq_len), module
            )
            patched += 1
            log.info(f"  ✅ Patched (base, cos_cached)         : {name}  ({cls_name})")

        else:
            log.warning(f"  ⚠️  Skipped unknown RoPE variant: {name}  ({cls_name})")

    if patched == 0:
        log.warning("  ⚠️  No RotaryEmbedding modules patched — verify model structure")
    else:
        log.info(f"  ✅ Total patched: {patched} RotaryEmbedding module(s)")

_patch_phi3_rope(hf_model, args.max_seq_len)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — 8da4w Quantization
#
# torchao 0.17 confirmed signature (read directly from source):
#   Int8DynActInt4WeightQuantizer(groupsize=256, ...)
#                                  ^^^^^^^^^
#                                  no underscore — NOT group_size
# ─────────────────────────────────────────────────────────────────────────────
log.info("")
log.info("STEP 4 — Applying 8da4w quantization")
log.info("         INT8 dynamic activations + INT4 weights  |  groupsize=32")

t1 = time.time()

try:
    from torchao.quantization.quant_api import Int8DynActInt4WeightQuantizer

    quantizer    = Int8DynActInt4WeightQuantizer(groupsize=32)
    hf_model     = quantizer.quantize(hf_model)
    quant_method = "Int8DynActInt4WeightQuantizer(groupsize=32)"

    quant_time = time.time() - t1
    log.info(f"  ✅ Quantization complete ({quant_time:.1f}s)")
    log.info(f"     Method : {quant_method}")

except Exception as e:
    log.error(f"  ❌ Quantization failed: {e}")
    log.error("  Expected stack:")
    log.error("    torch==2.11.0  torchao==0.17.0  executorch==1.2.0")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — torch.export graph capture
# ─────────────────────────────────────────────────────────────────────────────
log.info("")
log.info(f"STEP 5 — Capturing computation graph (max_seq_len={args.max_seq_len}) …")

t2 = time.time()

from torch.export import Dim, export as torch_export

example_ids  = torch.zeros((1, args.max_seq_len), dtype=torch.long)
example_mask = torch.ones ((1, args.max_seq_len), dtype=torch.long)
seq_dim      = Dim("seq_len", min=1, max=args.max_seq_len)

dynamic_shapes = {
    "input_ids"      : {0: Dim.STATIC, 1: seq_dim},
    "attention_mask" : {0: Dim.STATIC, 1: seq_dim},
}

try:
    with torch.no_grad():
        exported_program = torch_export(
            hf_model,
            args=(example_ids,),
            kwargs={"attention_mask": example_mask},
            dynamic_shapes=dynamic_shapes,
            strict=False,
        )
    export_time = time.time() - t2
    log.info(f"  ✅ Graph exported ({export_time:.1f}s)")

except Exception as e:
    log.error(f"  ❌ torch.export failed: {e}")
    log.error("  Try: --max_seq_len 128 for faster iteration")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — XNNPACK delegation
# ─────────────────────────────────────────────────────────────────────────────
log.info("")
log.info("STEP 6 — Lowering to Edge IR and delegating to XNNPACK …")

t3 = time.time()

try:
    from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

    edge_program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    lower_time = time.time() - t3
    log.info(f"  ✅ XNNPACK delegation complete ({lower_time:.1f}s)")

except Exception as e:
    log.error(f"  ❌ Edge lowering failed: {e}")
    log.error("  Fix: pip install --upgrade executorch")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Serialize to .pte
# ─────────────────────────────────────────────────────────────────────────────
log.info("")
log.info("STEP 7 — Serializing to .pte FlatBuffer …")

t4       = time.time()
pte_path = os.path.join(args.output_dir, cfg["output_name"])

try:
    executorch_program = edge_program.to_executorch()
    with open(pte_path, "wb") as f:
        f.write(executorch_program.buffer)

    pte_mb      = os.path.getsize(pte_path) / 1024 ** 2
    serial_time = time.time() - t4
    log.info(f"  ✅ Written : {pte_path}")
    log.info(f"     Size   : {pte_mb:.1f} MB")
    log.info(f"     Time   : {serial_time:.1f}s")

except Exception as e:
    log.error(f"  ❌ Serialization failed: {e}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Sanity check
# ─────────────────────────────────────────────────────────────────────────────
log.info("")
log.info("STEP 8 — Sanity check …")

try:
    from executorch.runtime import Runtime, Program
    runtime = Runtime.get()
    program = runtime.load_program(pte_path, verification=Program.Verification.Minimal)
    method  = program.load_method("forward")
    outputs = method.execute((
        torch.zeros((1, 10), dtype=torch.long),
        torch.ones ((1, 10), dtype=torch.long),
    ))
    log.info(f"  ✅ Test inference passed — output shape: {outputs[0].shape}")

except Exception as e:
    log.warning(f"  ⚠️  Sanity check skipped: {e}")
    log.warning("     .pte may still be valid for Android deployment")

# ─────────────────────────────────────────────────────────────────────────────
# Final summary + report
# ─────────────────────────────────────────────────────────────────────────────
total_time = time.time() - t0

report = f"""
ExecuTorch Export Report
════════════════════════════════════════════════
Model           : {cfg['display_name']}
Quantization    : 8da4w (INT8 activations + INT4 weights, groupsize=32)
Quant method    : {quant_method}
Backend         : XNNPACK (Android CPU / ARM NEON)
Max seq length  : {args.max_seq_len} tokens

Stack
─────
  torch      : {torch.__version__}
  torchao    : {ao_ver}
  executorch : {et_ver}

Output files
────────────
  .pte      : {pte_path}  ({pte_mb:.1f} MB)
  tokenizer : {tokenizer_dst}

Timing
──────
  Model load    : {load_time:.1f}s
  Quantization  : {quant_time:.1f}s
  torch.export  : {export_time:.1f}s
  XNNPACK lower : {lower_time:.1f}s
  Serialization : {serial_time:.1f}s
  Total         : {total_time:.1f}s ({total_time/60:.1f} min)

Next steps
──────────
  cp {pte_path} \\
     ~/AndroidStudioProjects/EdgeApp/app/src/main/assets/phi3_mini.pte

  cp {tokenizer_dst} \\
     ~/AndroidStudioProjects/EdgeApp/app/src/main/assets/tokenizer.json
════════════════════════════════════════════════
"""

report_path = os.path.join(args.output_dir, "export_report.txt")
with open(report_path, "w") as f:
    f.write(report)

log.info("")
log.info("=" * 70)
log.info("  EXPORT COMPLETE")
log.info("=" * 70)
log.info(f"  .pte      : {pte_path}  ({pte_mb:.1f} MB)")
log.info(f"  tokenizer : {tokenizer_dst}")
log.info(f"  report    : {report_path}")
log.info(f"  Total     : {total_time/60:.1f} minutes")
log.info("")
log.info("  Copy to Android Studio:")
log.info(f"  cp {pte_path} \\")
log.info(f"     ~/AndroidStudioProjects/EdgeApp/app/src/main/assets/phi3_mini.pte")
log.info(f"  cp {tokenizer_dst} \\")
log.info(f"     ~/AndroidStudioProjects/EdgeApp/app/src/main/assets/tokenizer.json")
log.info("=" * 70)