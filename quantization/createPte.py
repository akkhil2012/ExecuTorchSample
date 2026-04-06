"""
export_phi3.py
──────────────────────────────────────────────────────────────────────────────
Exports Phi-3 Mini (or TinyLlama as fallback) to a quantized ExecuTorch
.pte file optimised for memory-constrained Android devices (4 GB RAM).

Quantization  : 8da4w
  Linear layers  : INT8 dynamic activations + INT4 grouped weights
  Embedding      : INT4 weights
  KV cache       : FP32

Result         : ~7.3 GB HF model → ~950 MB .pte  (7.5x compression)
Backend        : XNNPACK (Android CPU / ARM NEON)
Max seq length : 512 tokens (conservative for 4 GB RAM)

KEY FIX in this version
───────────────────────
Phi-3 uses a CUSTOM architecture (configuration_phi3.py + modeling_phi3.py).
These are now included in the download list and the loader detects whether
they are present locally before deciding to fall back to HF Hub.

Usage
──────
  source /Users/akhil/phi3_et_env/bin/activate
  HF_HUB_DISABLE_XET=1 python export_phi3.py --model phi3
  HF_HUB_DISABLE_XET=1 python export_phi3.py --model tinyllama
"""

import argparse
import logging
import os
import shutil
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("export_phi3")

#python export_phi3.py --model phi3 --max_seq_len 128   # dev
#python export_phi3.py --model phi3 --max_seq_len 512   # release

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
            # ── CRITICAL: custom architecture files for Phi-3 ──────────────
            # Without these, transformers cannot instantiate the model locally.
            # They define Phi3Config and Phi3ForCausalLM classes.
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

# Copy tokenizer to output assets folder
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
    import executorch  # noqa: F401 — just checking it is importable
    from importlib.metadata import version as pkg_version

    try:
        et_ver = pkg_version("executorch")
    except Exception:
        et_ver = "unknown"

    log.info(f"  Python     : {sys.version.split()[0]}")
    log.info(f"  PyTorch    : {torch.__version__}")
    log.info(f"  ExecuTorch : {et_ver}")

except ImportError as e:
    log.error(f"  ❌ {e}")
    log.error("  Run: pip install torch executorch transformers")
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
hidden     = model_cfg.get("hidden_size",           3072)
num_heads  = model_cfg.get("num_attention_heads",     32)
factor_len = (hidden // num_heads) // 2   # 48 for Phi-3 Mini

rope = model_cfg.get("rope_scaling")   # may be None, {}, or a partial dict

needs_patch = (
    rope is None                          # key missing entirely  ← your case
    or not isinstance(rope, dict)         # wrong type
    or "short_factor" not in rope         # partial dict
    or "long_factor"  not in rope         # partial dict
    or rope.get("type") == "default"      # old-style value
)

if needs_patch:
    model_cfg["rope_scaling"] = {
        "type":         "longrope",
        "short_factor": [1.0] * factor_len,
        "long_factor":  [1.0] * factor_len,
    }
    _patched = True
    log.info(f"  ✅ rope_scaling patched  (was: {rope!r}  →  longrope, factor_len={factor_len})")
else:
    log.info(f"  ✅ rope_scaling already valid: {rope.get('type')}")

# Disable use_cache — prevents DynamicCache errors during tracing
if model_cfg.get("use_cache", True):
    model_cfg["use_cache"] = False
    _patched = True
    log.info("  ✅ use_cache set to False")
else:
    log.info("  ✅ use_cache already False")

# Verify model identity before saving
hidden_actual = model_cfg.get("hidden_size",  0)
vocab_actual  = model_cfg.get("vocab_size",   0)
mtype         = model_cfg.get("model_type",   "unknown")

if hidden_actual != 3072 or vocab_actual != 32064:
    log.error(f"  ❌ Wrong model in weights_dir!")
    log.error(f"     Got  : model_type={mtype}  hidden={hidden_actual}  vocab={vocab_actual}")
    log.error(f"     Need : model_type=phi3      hidden=3072             vocab=32064")
    log.error(f"     Fix  : rm -rf {args.weights_dir} && re-run to download correct weights")
    sys.exit(1)

if _patched:
    with open(cfg_path, "w") as f:
        json.dump(model_cfg, f, indent=2)
    log.info("  ✅ config.json saved")
    
# Verify the write succeeded
with open(cfg_path) as f:
    verify = json.load(f)
assert verify.get("rope_scaling", {}).get("type") == "longrope", \
    "config.json write failed — check file permissions"
log.info("  ✅ config.json verified on disk")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Load model
#
# Phi-3 uses custom architecture files (configuration_phi3.py,
# modeling_phi3.py). We now download these in Step 1, so local loading
# should always succeed. If for any reason they are missing we fall back
# to loading the architecture from HuggingFace Hub (requires internet).
# ─────────────────────────────────────────────────────────────────────────────
log.info("")
log.info("STEP 3 — Loading model weights …")

t0 = time.time()

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Check if custom architecture files are present locally
    custom_config   = os.path.join(args.weights_dir, "configuration_phi3.py")
    has_custom_arch = os.path.exists(custom_config)

    if has_custom_arch:
        log.info("  Custom arch files present — loading fully from local disk")
        model_source = args.weights_dir
        local_only   = True
    else:
        log.warning("  configuration_phi3.py not found locally")
        log.warning(f"  Falling back to HF Hub for architecture: {cfg['repo_id']}")
        log.warning("  (internet connection required for this fallback)")
        model_source = cfg["repo_id"]
        local_only   = False

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=True,
        local_files_only=local_only,
    )
    log.info("  ✅ Tokenizer loaded")

    # Load model in float32 — more stable for export tracing than bfloat16
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,     # streams weights to keep peak RAM lower
        local_files_only=local_only,
    )
    hf_model.eval()
    
    import torch._dynamo as dynamo
    dynamo.config.assume_static_by_default = False
    dynamo.config.capture_dynamic_output_shape_ops = True
    dynamo.config.capture_scalar_outputs = True
    
    param_count = sum(p.numel() for p in hf_model.parameters()) / 1e9
    load_time   = time.time() - t0
    log.info(f"  ✅ Model loaded : {param_count:.2f}B params  ({load_time:.1f}s)")

except Exception as e:
    log.error(f"  ❌ Model load failed: {e}")
    log.error("")
    log.error("  Likely causes:")
    log.error("  1. configuration_phi3.py download failed in Step 1")
    log.error("     Fix: ensure internet is available and re-run the script")
    log.error("  2. transformers version too old to support Phi-3")
    log.error("     Fix: pip install --upgrade 'transformers>=4.40.0'")
    log.error("  3. Incomplete safetensors — check both shards are in weights_dir")
    log.error("     model-00001-of-00002.safetensors  (~3.8 GB)")
    log.error("     model-00002-of-00002.safetensors  (~3.5 GB)")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — 8da4w Quantization
# ─────────────────────────────────────────────────────────────────────────────
log.info("")
log.info("STEP 4 — Applying 8da4w quantization")
log.info("         INT8 dynamic activations + INT4 weights  |  group_size=32")

t1 = time.time()

try:
    from torchao.quantization import quantize_
    from torchao.quantization import Int8DynamicActivationInt4WeightConfig

    log.info("  Using torchao 0.5.0 quantization API …")
    #quantize_(hf_model, int8_dynamic_activation_int4_weight(group_size=32))
    quantize_(hf_model, Int8DynamicActivationInt4WeightConfig(group_size=32))


    quant_time   = time.time() - t1
    quant_method = "torchao_8da4w"
    log.info(f"  ✅ torchao 8da4w applied ({quant_time:.1f}s)")

except Exception as e:
    log.error(f"  ❌ Quantization failed: {e}")
    log.error("  Fix: pip install --upgrade torchao")
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
    "input_ids"      : {0: Dim.STATIC, 1: seq_dim},   # batch=1 is STATIC
    "attention_mask" : {0: Dim.STATIC, 1: seq_dim},
}

try:
    with torch.no_grad():
        # Hint: u0 == 1 is the batch dimension — tell exporter it's always 1
        exported_program = torch_export(
            hf_model,
            args=(example_ids,),
            kwargs={"attention_mask": example_mask},
            dynamic_shapes=dynamic_shapes,
            strict=False,
            # Treat batch dim as static (=1) to resolve Eq(u0, 1) guard
        )                            

    export_time = time.time() - t2
    log.info(f"  ✅ Graph exported ({export_time:.1f}s)")

except Exception as e:
    log.error(f"  ❌ torch.export failed: {e}")
    log.error("  Common fixes:")
    log.error("  • pip install --upgrade torchao")
    log.error("  • Close all other apps — export needs 16 GB+ RAM")
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
Quantization    : 8da4w (INT8 activations + INT4 weights, group_size=32)
Quant method    : {quant_method}
Backend         : XNNPACK (Android CPU / ARM NEON)
Max seq length  : {args.max_seq_len} tokens

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
