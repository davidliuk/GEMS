#!/usr/bin/env python3
"""
Auto-download model checkpoints and benchmark datasets for ComfyClaw experiments.

Usage:
    python experiments/setup.py --all                          # everything
    python experiments/setup.py --model longcat --model qwen   # specific models
    python experiments/setup.py --benchmark geneval2            # specific benchmark
    python experiments/setup.py --models-only                  # all models, no benchmarks
    python experiments/setup.py --benchmarks-only              # all benchmarks, no models
    python experiments/setup.py --dry-run --all                # show what would download
"""
import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [setup] %(message)s",
)
log = logging.getLogger("setup")

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Model download definitions ────────────────────────────────────────────
# Each entry: (hf_repo, hf_path, local_subdir_under_comfyui_models)
# Files shared across models (e.g. ae.safetensors) are deduplicated at runtime.

MODEL_FILES: dict[str, list[tuple[str, str, str]]] = {
    "longcat": [
        ("Comfy-Org/LongCat-Image",
         "split_files/diffusion_models/longcat_image_bf16.safetensors",
         "diffusion_models"),
        ("Comfy-Org/Qwen-Image_ComfyUI",
         "split_files/text_encoders/qwen_2.5_vl_7b.safetensors",
         "text_encoders"),
        ("f5aiteam/ComfyUI",
         "models/vae/ae.safetensors",
         "vae"),
    ],
    "qwen": [
        ("Comfy-Org/Qwen-Image_ComfyUI",
         "split_files/diffusion_models/qwen_image_bf16.safetensors",
         "diffusion_models"),
        ("Comfy-Org/Qwen-Image_ComfyUI",
         "split_files/text_encoders/qwen_2.5_vl_7b.safetensors",
         "text_encoders"),
        ("Comfy-Org/Qwen-Image_ComfyUI",
         "split_files/vae/qwen_image_vae.safetensors",
         "vae"),
    ],
    "z-image-turbo": [
        ("Comfy-Org/z_image_turbo",
         "split_files/diffusion_models/z_image_turbo_bf16.safetensors",
         "diffusion_models"),
        ("Comfy-Org/z_image_turbo",
         "split_files/text_encoders/qwen_3_4b.safetensors",
         "text_encoders"),
        ("f5aiteam/ComfyUI",
         "models/vae/ae.safetensors",
         "vae"),
    ],
    "dreamshaper": [
        ("Lykon/DreamShaper",
         "DreamShaper_8_pruned.safetensors",
         "checkpoints"),
    ],
}

# ── Benchmark download definitions ────────────────────────────────────────
# method: "git" for git clone, "hf_dataset" for huggingface datasets download

BENCHMARK_SOURCES: dict[str, dict] = {
    "geneval2": {
        "method": "git",
        "url": "https://github.com/facebookresearch/GenEval2.git",
        "target_dir": "GenEval2",
    },
    "dpg-bench": {
        "method": "hf_dataset",
        "repo_id": "Jialuo21/DPG-Bench",
        "target_dir": "DPG-Bench",
    },
    "oneig-en": {
        "method": "hf_dataset",
        "repo_id": "OneIG-Bench/OneIG-Bench",
        "target_dir": "OneIG-Bench",
        "files": ["OneIG-Bench.json"],
    },
    "oneig-zh": {
        "method": "hf_dataset",
        "repo_id": "OneIG-Bench/OneIG-Bench",
        "target_dir": "OneIG-Bench",
        "files": ["OneIG-Bench-ZH.json"],
    },
    "wise": {
        "method": "git",
        "url": "https://github.com/PKU-YuanGroup/WISE.git",
        "target_dir": "WISE",
    },
}


def _file_basename(hf_path: str) -> str:
    return hf_path.rsplit("/", 1)[-1]


def download_model_files(
    model: str,
    comfyui_dir: Path,
    dry_run: bool = False,
) -> None:
    import tempfile

    from huggingface_hub import hf_hub_download

    files = MODEL_FILES[model]
    log.info("Model: %s (%d file(s))", model, len(files))

    for repo_id, hf_path, subdir in files:
        filename = _file_basename(hf_path)
        dest_dir = comfyui_dir / "models" / subdir
        dest_path = dest_dir / filename

        if dest_path.exists():
            size_mb = dest_path.stat().st_size / (1024 * 1024)
            log.info("  SKIP %s (already exists, %.0f MB)", dest_path, size_mb)
            continue

        if dry_run:
            log.info("  WOULD DOWNLOAD %s → %s", f"{repo_id}/{hf_path}", dest_path)
            continue

        log.info("  Downloading %s from %s ...", filename, repo_id)
        dest_dir.mkdir(parents=True, exist_ok=True)

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    filename=hf_path,
                    local_dir=tmp_dir,
                )
                shutil.move(str(downloaded), str(dest_path))

            size_mb = dest_path.stat().st_size / (1024 * 1024)
            log.info("  OK %s (%.0f MB)", dest_path, size_mb)
        except Exception as exc:
            if dest_path.exists():
                dest_path.unlink()
            log.error("  FAILED %s: %s", filename, exc)
            raise


def download_benchmark(
    benchmark: str,
    data_root: Path,
    dry_run: bool = False,
) -> None:
    src = BENCHMARK_SOURCES[benchmark]
    target = data_root / src["target_dir"]

    log.info("Benchmark: %s", benchmark)

    if src["method"] == "git":
        if target.exists():
            log.info("  SKIP %s (already exists)", target)
            return
        if dry_run:
            log.info("  WOULD CLONE %s → %s", src["url"], target)
            return

        log.info("  Cloning %s ...", src["url"])
        subprocess.run(
            ["git", "clone", "--depth", "1", src["url"], str(target)],
            check=True,
        )
        log.info("  OK %s", target)

    elif src["method"] == "hf_dataset":
        from huggingface_hub import hf_hub_download

        target.mkdir(parents=True, exist_ok=True)
        specific_files = src.get("files")

        if specific_files:
            for fname in specific_files:
                dest = target / fname
                if dest.exists():
                    log.info("  SKIP %s (already exists)", dest)
                    continue
                if dry_run:
                    log.info("  WOULD DOWNLOAD %s/%s → %s", src["repo_id"], fname, dest)
                    continue

                log.info("  Downloading %s ...", fname)
                downloaded = hf_hub_download(
                    repo_id=src["repo_id"],
                    filename=fname,
                    repo_type="dataset",
                    local_dir=str(target),
                )
                log.info("  OK %s", downloaded)
        else:
            marker = target / ".downloaded"
            if marker.exists():
                log.info("  SKIP %s (already downloaded)", target)
                return
            if dry_run:
                log.info("  WOULD DOWNLOAD dataset %s → %s", src["repo_id"], target)
                return

            from huggingface_hub import snapshot_download

            log.info("  Downloading dataset %s ...", src["repo_id"])
            snapshot_download(
                repo_id=src["repo_id"],
                repo_type="dataset",
                local_dir=str(target),
            )
            marker.write_text("done\n")
            log.info("  OK %s", target)


def main():
    all_models = list(MODEL_FILES.keys())
    all_benchmarks = list(BENCHMARK_SOURCES.keys())

    parser = argparse.ArgumentParser(
        description="Download model checkpoints and benchmark datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"Models:     {', '.join(all_models)}\n"
            f"Benchmarks: {', '.join(all_benchmarks)}"
        ),
    )
    parser.add_argument("--model", action="append", choices=all_models, default=None,
                        help="Model(s) to download (repeatable)")
    parser.add_argument("--benchmark", action="append", choices=all_benchmarks, default=None,
                        help="Benchmark(s) to download (repeatable)")
    parser.add_argument("--all", action="store_true",
                        help="Download all models and benchmarks")
    parser.add_argument("--models-only", action="store_true",
                        help="Download all models (no benchmarks)")
    parser.add_argument("--benchmarks-only", action="store_true",
                        help="Download all benchmarks (no models)")
    parser.add_argument("--comfyui-dir", type=str, default="/workspace/ComfyUI",
                        help="ComfyUI installation path (default: /workspace/ComfyUI)")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Root directory for benchmark data (default: parent of repo)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be downloaded without downloading")
    args = parser.parse_args()

    comfyui_dir = Path(args.comfyui_dir)
    data_root = Path(args.data_root) if args.data_root else REPO_ROOT.parent

    if args.all:
        models_to_dl = all_models
        benchmarks_to_dl = all_benchmarks
    elif args.models_only:
        models_to_dl = all_models
        benchmarks_to_dl = args.benchmark or []
    elif args.benchmarks_only:
        models_to_dl = args.model or []
        benchmarks_to_dl = all_benchmarks
    elif args.model or args.benchmark:
        models_to_dl = args.model or []
        benchmarks_to_dl = args.benchmark or []
    else:
        parser.print_help()
        print("\nSpecify --all, --models-only, --benchmarks-only, or specific --model/--benchmark flags.")
        sys.exit(1)

    if args.dry_run:
        log.info("DRY RUN — no files will be downloaded")

    log.info("=" * 60)
    log.info("ComfyClaw Setup")
    log.info("=" * 60)
    log.info("ComfyUI dir:  %s", comfyui_dir)
    log.info("Data root:    %s", data_root)
    log.info("Models:       %s", ", ".join(models_to_dl) or "(none)")
    log.info("Benchmarks:   %s", ", ".join(benchmarks_to_dl) or "(none)")
    log.info("=" * 60)

    if not comfyui_dir.exists() and models_to_dl:
        log.error("ComfyUI directory not found: %s", comfyui_dir)
        log.error("Set --comfyui-dir to your ComfyUI installation path")
        sys.exit(1)

    # ── Download models ──────────────────────────────────────────────
    if models_to_dl:
        log.info("")
        log.info("── Model Checkpoints ──")
        for model in models_to_dl:
            download_model_files(model, comfyui_dir, dry_run=args.dry_run)
            log.info("")

    # ── Download benchmarks ──────────────────────────────────────────
    if benchmarks_to_dl:
        log.info("")
        log.info("── Benchmark Datasets ──")
        for bench in benchmarks_to_dl:
            download_benchmark(bench, data_root, dry_run=args.dry_run)
            log.info("")

    log.info("=" * 60)
    log.info("Setup complete!")
    if models_to_dl:
        log.info("Restart ComfyUI to pick up new models.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
