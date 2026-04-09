"""
comfyclaw CLI — entry point installed as the ``comfyclaw`` script.

Configuration precedence (highest → lowest):
  1. Explicit CLI flags
  2. Environment variables  (can be loaded from a .env file)
  3. Built-in defaults

All sensitive configuration (API key, paths) is read from environment
variables only — never hardcoded.  Copy ``.env.example`` to ``.env`` and
fill in your values, or export the variables directly.

Sub-commands
------------
run          Run the full agent–generate–verify loop.
dry-run      Run the agent only (no ComfyUI execution needed).
install-node Symlink the ComfyClaw-Sync custom node into ComfyUI.
node-path    Print the path to the bundled custom node directory.

Environment variables
---------------------
ANTHROPIC_API_KEY        Required for run / dry-run.
COMFYUI_DIR              Path to ComfyUI installation (install-node).
COMFYUI_ADDR             host:port of a running ComfyUI server.
COMFYCLAW_MODEL          Claude model name.
COMFYCLAW_MAX_ITERATIONS Max agent–generate–verify cycles.
COMFYCLAW_THRESHOLD      Stop early when verifier score ≥ this.
COMFYCLAW_SCORE_WEIGHTS  Comma-separated "req_w,detail_w" (sum=1).
COMFYCLAW_EVOLVE_FROM_BEST  "true"/"false" for topology accumulation.
COMFYCLAW_SYNC_PORT      WebSocket port (0 = disable).
COMFYCLAW_SKILLS_DIR     Custom skills directory path.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# .env loader — runs at import time so env vars are available everywhere
# ─────────────────────────────────────────────────────────────────────────────

def _load_dotenv() -> None:
    """
    Load `.env` from the current working directory or the package root.
    Silently skips if python-dotenv is not installed or no .env exists.
    Existing environment variables are NOT overwritten.
    """
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]
    except ImportError:
        return  # python-dotenv is optional

    # Look in cwd first, then package root
    cwd_env = Path.cwd() / ".env"
    pkg_env = Path(__file__).resolve().parent.parent / ".env"
    env_path = cwd_env if cwd_env.exists() else (pkg_env if pkg_env.exists() else None)
    if env_path:
        load_dotenv(env_path, override=False)


_load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# Typed config helpers
# ─────────────────────────────────────────────────────────────────────────────

def _require_env(name: str, hint: str = "") -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        msg = f"Error: {name!r} is not set."
        if hint:
            msg += f"\n{hint}"
        sys.exit(msg)
    return val


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        print(f"[cli] Warning: {name}={raw!r} is not an integer, using {default}.", file=sys.stderr)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        print(f"[cli] Warning: {name}={raw!r} is not a float, using {default}.", file=sys.stderr)
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_score_weights(default: tuple[float, float] = (0.6, 0.4)) -> tuple[float, float]:
    raw = os.environ.get("COMFYCLAW_SCORE_WEIGHTS", "").strip()
    if not raw:
        return default
    try:
        parts = [float(x) for x in raw.split(",")]
        if len(parts) == 2:
            return (parts[0], parts[1])
    except ValueError:
        pass
    print(f"[cli] Warning: COMFYCLAW_SCORE_WEIGHTS={raw!r} invalid, using {default}.", file=sys.stderr)
    return default


# ─────────────────────────────────────────────────────────────────────────────
# Derived defaults
# ─────────────────────────────────────────────────────────────────────────────

def _api_key() -> str:
    return _require_env(
        "ANTHROPIC_API_KEY",
        hint="Export it before running:\n  export ANTHROPIC_API_KEY=sk-ant-...\n"
             "Or add it to your .env file (see .env.example).",
    )


def _comfyui_dir() -> Path:
    raw = os.environ.get("COMFYUI_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / "Documents" / "ComfyUI"


def _server_addr() -> str:
    return _env_str("COMFYUI_ADDR", "127.0.0.1:8188")


def _bundled_custom_node() -> Path:
    """Return the path to the ComfyClaw-Sync custom node bundled inside the package."""
    pkg_node = Path(__file__).resolve().parent / "custom_node"
    if pkg_node.is_dir():
        return pkg_node
    # Development / repo layout
    repo_node = Path(__file__).resolve().parent.parent.parent / "custom_nodes" / "ComfyClaw-Sync"
    return repo_node


# ─────────────────────────────────────────────────────────────────────────────
# Custom-node management
# ─────────────────────────────────────────────────────────────────────────────

def _install_node(comfyui_dir: Path) -> None:
    """Symlink the ComfyClaw-Sync custom node into ComfyUI's custom_nodes/."""
    src = _bundled_custom_node()
    dst = comfyui_dir / "custom_nodes" / "ComfyClaw-Sync"

    if dst.exists() or dst.is_symlink():
        print(f"[cli] Custom node already installed at {dst}")
        return
    if not (comfyui_dir / "custom_nodes").exists():
        print(
            f"[cli] ⚠  ComfyUI custom_nodes dir not found at {comfyui_dir}.\n"
            "       Set COMFYUI_DIR in .env or pass --comfyui-dir."
        )
        return
    if not src.exists():
        print(f"[cli] ⚠  ComfyClaw-Sync source not found at {src}.")
        return
    try:
        dst.symlink_to(src.resolve())
        print(f"[cli] ✅ Symlinked {src.resolve()} → {dst}")
        print("       Restart ComfyUI to activate the sync extension.")
    except Exception as exc:
        print(f"[cli] ❌ Symlink failed: {exc}")
        print(f"       Manual install:\n  cp -r {src.resolve()} {dst.parent}/")


# ─────────────────────────────────────────────────────────────────────────────
# ComfyUI startup helper
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_comfyui_running(addr: str) -> str:
    """Ping ComfyUI; try to open the Desktop app if not responding."""
    from .client import ComfyClient
    client = ComfyClient(addr)
    if client.is_alive():
        print(f"[cli] ComfyUI is UP at http://{addr}")
        return addr

    print("[cli] ComfyUI not running — attempting to open the app…")
    app_path = Path("/Applications/ComfyUI.app")
    try:
        subprocess.Popen(
            ["open", str(app_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        print(f"[cli] Could not open ComfyUI: {exc}")
        return addr

    print("[cli] Waiting up to 60 s for ComfyUI to start…")
    from .client import ComfyClient as CC
    for port in (8188, 8000):
        probe_addr = f"127.0.0.1:{port}"
        probe = CC(probe_addr)
        for _ in range(30):
            time.sleep(2)
            if probe.is_alive():
                print(f"[cli] ComfyUI started at http://{probe_addr}")
                return probe_addr
    return addr


def _save_image(image_bytes: bytes, prompt: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = prompt[:40].replace(" ", "_").replace("/", "-")
    ts = int(time.time())
    out = output_dir / f"comfyclaw_{ts}_{slug}.png"
    out.write_bytes(image_bytes)
    print(f"[cli] Image saved → {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Sub-command handlers
# ─────────────────────────────────────────────────────────────────────────────

def _cmd_run(args: argparse.Namespace, dry: bool = False) -> None:
    from .harness import ClawHarness, HarnessConfig

    api_key = _api_key()
    addr = _server_addr()

    if not dry:
        addr = _ensure_comfyui_running(addr)

    # CLI flags override env vars; env vars already loaded as defaults
    cfg = HarnessConfig(
        api_key=api_key,
        server_address=addr,
        model=args.model,
        max_iterations=args.iterations,
        success_threshold=args.threshold,
        sync_port=0 if args.no_sync else args.sync_port,
        skills_dir=args.skills_dir,
        evolve_from_best=not args.reset_each_iter,
        score_weights=_env_score_weights(),
        image_model=args.image_model or None,
    )

    print(f"\n[cli] Workflow    : {args.workflow}")
    print(f"[cli] Prompt      : {args.prompt!r}")
    print(f"[cli] Model       : {cfg.model}")
    print(f"[cli] Image model : {cfg.image_model or '(from workflow)'}")
    print(f"[cli] Iterations  : {cfg.max_iterations}  Threshold: {cfg.success_threshold}")
    print(f"[cli] Dry-run     : {dry}")
    print(f"[cli] Sync port   : {cfg.sync_port or 'disabled'}")
    print(f"[cli] Evolve mode : {'accumulate' if cfg.evolve_from_best else 'reset'}")

    with ClawHarness.from_workflow_file(args.workflow, cfg) as h:
        result = h.run(prompt=args.prompt, dry_run=dry)

    if result:
        out_dir = Path(args.output_dir) if args.output_dir else Path.cwd() / "comfyclaw_output"
        _save_image(result, args.prompt, out_dir)
    elif dry:
        print("\n[cli] Dry-run complete.")


def _cmd_install_node(args: argparse.Namespace) -> None:
    comfyui_dir = Path(args.comfyui_dir).expanduser() if args.comfyui_dir else _comfyui_dir()
    _install_node(comfyui_dir)


def _cmd_node_path(_args: argparse.Namespace) -> None:
    """Print the path to the bundled ComfyClaw-Sync custom node directory."""
    print(_bundled_custom_node())


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="comfyclaw",
        description="ComfyClaw — agentic self-evolving ComfyUI harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Env-var configuration: copy .env.example → .env and fill in your values.\n"
            "All CLI flags override the corresponding env var.\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def _add_run_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--workflow", required=True, metavar="PATH",
                       help="Path to API-format ComfyUI workflow JSON")
        p.add_argument("--prompt", required=True,
                       help="Image generation prompt")
        p.add_argument("--model",
                       default=_env_str("COMFYCLAW_MODEL", "claude-sonnet-4-5"),
                       metavar="NAME")
        p.add_argument("--iterations", type=int,
                       default=_env_int("COMFYCLAW_MAX_ITERATIONS", 3),
                       metavar="N")
        p.add_argument("--threshold", type=float,
                       default=_env_float("COMFYCLAW_THRESHOLD", 0.85),
                       metavar="SCORE")
        p.add_argument("--sync-port", type=int,
                       default=_env_int("COMFYCLAW_SYNC_PORT", 8765),
                       metavar="PORT")
        p.add_argument("--no-sync", action="store_true",
                       help="Disable live WebSocket sync")
        p.add_argument("--skills-dir",
                       default=os.environ.get("COMFYCLAW_SKILLS_DIR") or None,
                       metavar="DIR")
        p.add_argument("--reset-each-iter", action="store_true",
                       default=not _env_bool("COMFYCLAW_EVOLVE_FROM_BEST", True),
                       help="Disable topology accumulation (reset to base each iteration)")
        p.add_argument("--output-dir", default=None, metavar="DIR",
                       help="Directory for saved output images")
        p.add_argument(
            "--image-model",
            default=os.environ.get("COMFYCLAW_IMAGE_MODEL", "").strip() or None,
            metavar="NAME",
            help=(
                "Pin the ComfyUI checkpoint / UNET to this model name, e.g. "
                "'Qwen/Qwen-Image-2512' or 'realisticVisionV51.safetensors'. "
                "Overrides COMFYCLAW_IMAGE_MODEL env var. "
                "Leave unset to use whatever model the workflow already specifies."
            ),
        )

    run_p = sub.add_parser("run", help="Run the full agent–generate–verify loop")
    _add_run_args(run_p)
    run_p.set_defaults(func=lambda a: _cmd_run(a, dry=False))

    dry_p = sub.add_parser("dry-run", help="Run agent only (no ComfyUI execution)")
    _add_run_args(dry_p)
    dry_p.set_defaults(func=lambda a: _cmd_run(a, dry=True))

    inst_p = sub.add_parser("install-node",
                             help="Symlink ComfyClaw-Sync custom node into ComfyUI")
    inst_p.add_argument("--comfyui-dir", default=None, metavar="DIR",
                        help="ComfyUI installation directory (or set COMFYUI_DIR in .env)")
    inst_p.set_defaults(func=_cmd_install_node)

    np_p = sub.add_parser("node-path",
                           help="Print path to the bundled ComfyClaw-Sync plugin")
    np_p.set_defaults(func=_cmd_node_path)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
