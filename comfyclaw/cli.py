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
Provider API keys (set the one matching your chosen --model provider):
  ANTHROPIC_API_KEY    Anthropic Claude  (default provider)
  OPENAI_API_KEY       OpenAI GPT-4o / o-series
  GEMINI_API_KEY       Google Gemini
  GROQ_API_KEY         Groq
  (none needed)        Local Ollama

COMFYUI_DIR              Path to ComfyUI installation (install-node).
COMFYUI_ADDR             host:port of a running ComfyUI server.
COMFYCLAW_MODEL          LiteLLM model string for the agent
                         (default: anthropic/claude-sonnet-4-5).
COMFYCLAW_VERIFIER_MODEL LiteLLM model for the vision verifier
                         (default: same as COMFYCLAW_MODEL).
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
    print(
        f"[cli] Warning: COMFYCLAW_SCORE_WEIGHTS={raw!r} invalid, using {default}.", file=sys.stderr
    )
    return default


# ─────────────────────────────────────────────────────────────────────────────
# Derived defaults
# ─────────────────────────────────────────────────────────────────────────────


def _api_key() -> str:
    """Return an API key from the environment.

    LiteLLM reads provider keys from env-vars automatically (ANTHROPIC_API_KEY,
    OPENAI_API_KEY, GEMINI_API_KEY, …).  We read ANTHROPIC_API_KEY here for
    backward compatibility when the model is Anthropic.  If you use a different
    provider, set that provider's env-var instead and leave ANTHROPIC_API_KEY
    unset — in that case this returns an empty string, which is fine.
    """
    return os.environ.get("ANTHROPIC_API_KEY", "").strip()


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


_LOCAL_HOSTS = {"127.0.0.1", "localhost", "0.0.0.0", "::1"}


def _is_local(host: str) -> bool:
    return host in _LOCAL_HOSTS


def _ensure_comfyui_running(addr: str) -> str:
    """Ping ComfyUI; auto-discover port or launch Desktop app if local."""
    from .client import ComfyClient

    client = ComfyClient(addr)
    if client.is_alive():
        print(f"[cli] ComfyUI is UP at http://{addr}")
        return addr

    host = addr.split(":")[0] if ":" in addr else "127.0.0.1"

    # Scan common ports on the same host
    probe_ports = [8188, 8000, 8080, 7130]
    for port in probe_ports:
        alt = f"{host}:{port}"
        if alt != addr and ComfyClient(alt).is_alive():
            print(f"[cli] ComfyUI found at http://{alt}")
            return alt

    # Remote host: nothing more we can do — just warn and proceed
    if not _is_local(host):
        print(f"[cli] ⚠  ComfyUI not responding at {addr}")
        print(f"[cli]    Verify ComfyUI is running on the remote host and the address is correct.")
        print(f"[cli]    Proceeding with {addr} — the agent will fail if ComfyUI is unreachable.")
        return addr

    # Local host: try to launch the Desktop app (macOS)
    print("[cli] ComfyUI not responding locally — attempting to open the app…")
    app_path = Path("/Applications/ComfyUI.app")
    if not app_path.exists():
        print(f"[cli] ⚠  ComfyUI Desktop app not found at {app_path}")
        print(f"[cli]    Start ComfyUI manually, then re-run this command.")
        return addr
    try:
        subprocess.Popen(
            ["open", str(app_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        print(f"[cli] Could not open ComfyUI: {exc}")
        return addr

    print("[cli] Waiting up to 60 s for ComfyUI to start…")
    probe_addrs = [f"{host}:{p}" for p in probe_ports]
    for _ in range(30):
        time.sleep(2)
        for pa in probe_addrs:
            if ComfyClient(pa).is_alive():
                print(f"[cli] ComfyUI started at http://{pa}")
                return pa
    print("[cli] ⚠  Timed out waiting for ComfyUI. Proceeding with {addr}.")
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
    addr = args.comfyui_addr

    if not dry:
        addr = _ensure_comfyui_running(addr)

    # CLI flags override env vars; env vars already loaded as defaults
    verifier_model = args.verifier_model.strip() or None
    cfg = HarnessConfig(
        api_key=api_key,
        server_address=addr,
        model=args.model,
        verifier_model=verifier_model,
        max_iterations=args.iterations,
        success_threshold=args.threshold,
        sync_port=0 if args.no_sync else args.sync_port,
        skills_dir=args.skills_dir,
        evolve_from_best=not args.reset_each_iter,
        score_weights=_env_score_weights(),
        image_model=args.image_model or None,
        max_repair_attempts=args.max_repair_attempts,
        verifier_mode=args.verifier_mode,
    )

    verifier_label = cfg.verifier_model or f"{cfg.model} (shared)"
    print(f"\n[cli] Workflow       : {args.workflow or '(empty — agent builds from scratch)'}")
    print(f"[cli] Prompt         : {args.prompt!r}")
    print(f"[cli] Agent model    : {cfg.model}")
    print(f"[cli] Verifier mode  : {cfg.verifier_mode}")
    if cfg.verifier_mode in ("vlm", "hybrid"):
        print(f"[cli] Verifier model : {verifier_label}")
    print(f"[cli] Image model    : {cfg.image_model or '(from workflow)'}")
    print(f"[cli] Iterations     : {cfg.max_iterations}  Threshold: {cfg.success_threshold}")
    print(f"[cli] Dry-run        : {dry}")
    print(f"[cli] Sync port      : {cfg.sync_port or 'disabled'}")
    print(f"[cli] Evolve mode    : {'accumulate' if cfg.evolve_from_best else 'reset'}")
    print(f"[cli] Repair limit   : {cfg.max_repair_attempts} attempt(s) per iteration")

    if args.workflow:
        ctx = ClawHarness.from_workflow_file(args.workflow, cfg)
    else:
        ctx = ClawHarness.from_workflow_dict({}, cfg)
    with ctx as h:
        result = h.run(prompt=args.prompt, dry_run=dry)

    if result:
        out_dir = Path(args.output_dir) if args.output_dir else Path.cwd() / "comfyclaw_output"
        _save_image(result, args.prompt, out_dir)
    elif dry:
        print("\n[cli] Dry-run complete.")


def _cmd_serve(args: argparse.Namespace) -> None:
    """Persistent server mode — waits for trigger_generation from ComfyUI."""
    from .harness import ClawHarness, HarnessConfig
    from .sync_server import SyncServer

    api_key = _api_key()
    addr = _ensure_comfyui_running(args.comfyui_addr)

    sync_port = 0 if args.no_sync else args.sync_port
    if not sync_port:
        sys.exit("[cli] Error: serve mode requires sync (WebSocket). Do not use --no-sync.")

    verifier_model = args.verifier_model.strip() or None
    base_cfg = dict(
        api_key=api_key,
        server_address=addr,
        model=args.model,
        verifier_model=verifier_model,
        success_threshold=args.threshold,
        sync_port=0,  # harness won't create its own SyncServer; we inject the shared one
        skills_dir=args.skills_dir,
        evolve_from_best=not args.reset_each_iter,
        score_weights=_env_score_weights(),
        image_model=args.image_model or None,
        max_repair_attempts=args.max_repair_attempts,
    )

    print(f"\n[cli] ComfyClaw serve mode")
    print(f"[cli] Agent model    : {args.model}")
    print(f"[cli] Image model    : {base_cfg['image_model'] or '(from workflow)'}")
    print(f"[cli] Sync port      : {sync_port}")
    print(f"[cli] Waiting for triggers from ComfyUI panel…\n")

    sync = SyncServer(port=sync_port)
    sync.start()
    if not sync.is_running():
        print(f"[cli] Port {sync_port} appears busy — attempting to reclaim…")
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{sync_port}"],
                capture_output=True, text=True, timeout=5,
            )
            pids = result.stdout.strip().split()
            for pid in pids:
                if pid.isdigit() and int(pid) != os.getpid():
                    os.kill(int(pid), 9)
                    print(f"[cli] Killed stale process {pid}")
            if pids:
                time.sleep(1)
                sync = SyncServer(port=sync_port)
                sync.start()
        except Exception:
            pass
        if not sync.is_running():
            sys.exit(f"[cli] Error: SyncServer failed to start on port {sync_port}. "
                     f"Free the port manually: lsof -ti :{sync_port} | xargs kill")

    try:
        while True:
            print("[serve] ⏳ Waiting for generation trigger from ComfyUI…")
            trigger = sync.wait_for_trigger(timeout=0)
            if trigger is None:
                continue

            prompt = trigger.get("prompt", "").strip()
            if not prompt:
                sync.send_error("No prompt provided.")
                continue

            mode = trigger.get("mode", "scratch")
            settings = trigger.get("settings") or {}
            workflow = trigger.get("workflow") if mode == "improve" else {}
            if workflow is None:
                workflow = {}

            iterations = settings.get("iterations", args.iterations)
            verifier_mode = settings.get("verifier_mode", args.verifier_mode)

            # Model / API-key overrides from the ComfyUI panel
            trigger_model = settings.get("model", "").strip()
            trigger_api_key = settings.get("api_key", "").strip()
            trigger_verifier_model = settings.get("verifier_model", "").strip()

            run_cfg = dict(base_cfg)
            if trigger_model:
                run_cfg["model"] = trigger_model
            if trigger_api_key:
                run_cfg["api_key"] = trigger_api_key
            if trigger_verifier_model:
                run_cfg["verifier_model"] = trigger_verifier_model

            cfg = HarnessConfig(
                **run_cfg,
                max_iterations=iterations,
                verifier_mode=verifier_mode,
            )

            mode_label = "from scratch" if mode == "scratch" else "improve current"
            node_count = len(workflow) if workflow else 0
            print(f"\n[serve] 🚀 Trigger received: {prompt[:80]!r}")
            print(f"[serve]    Mode: {mode_label}, Iterations: {iterations}, "
                  f"Verifier: {verifier_mode}, Model: {cfg.model}, Nodes: {node_count}")

            sync.send_status("running", iteration=0, detail="Initializing agent…")

            if mode == "scratch":
                sync.reset()
                sync.broadcast({})

            try:
                harness = ClawHarness.from_workflow_dict(workflow, cfg)
                harness._sync = sync
                harness._agent.on_change = harness._on_workflow_change

                def _status_cb(state: str, iteration: int = 0, detail: str = "") -> None:
                    sync.send_status(state, iteration, detail)
                harness.on_status = _status_cb

                result = harness.run(prompt=prompt)

                if result:
                    out_dir = (
                        Path(args.output_dir) if args.output_dir
                        else Path.cwd() / "comfyclaw_output"
                    )
                    saved = _save_image(result, prompt, out_dir)
                    sync.send_complete(
                        score=0.0,
                        iterations_used=iterations,
                        image_path=str(saved),
                    )
                else:
                    sync.send_complete(score=0.0, iterations_used=0, image_path="")
            except Exception as exc:
                print(f"[serve] ❌ Error: {exc}")
                sync.send_error(str(exc))

    except KeyboardInterrupt:
        print("\n[serve] Shutting down…")
    finally:
        sync.stop()


def _cmd_benchmark(args: argparse.Namespace) -> None:
    """Run a T2I benchmark suite."""
    import logging

    from .benchmark import BenchmarkConfig, BenchmarkRunner

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    config = BenchmarkConfig(
        suite=args.suite,
        name=args.name,
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_iterations=args.max_iterations,
        num_workers=args.num_workers,
        server_address=args.comfyui_addr,
        api_key=_api_key(),
        model=args.model,
        image_model=args.image_model,
        workflow_path=args.workflow,
        stage_gated=args.stage_gated,
        verifier_model=args.verifier_model,
        max_prompts=args.max_prompts,
    )

    runner = BenchmarkRunner(config)
    result = runner.run()

    print("\n" + "=" * 50)
    print(result.summary())
    print("=" * 50)


def _cmd_evolve(args: argparse.Namespace) -> None:
    """Run a self-evolution cycle on benchmark results."""
    import json
    import logging

    from .evolve import SkillEvolver

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    with open(args.results, "r", encoding="utf-8") as f:
        results = json.load(f)

    evolved_dir = args.evolved_skills_dir or str(
        Path(__file__).resolve().parent / "skills_evolved"
    )

    evolver = SkillEvolver(
        evolved_skills_dir=evolved_dir,
        llm_model=args.model,
        api_key=_api_key(),
        min_improvement=args.min_improvement,
        max_mutations_per_cycle=args.max_mutations,
    )

    reports = evolver.run_multi_cycle(
        results=results,
        max_cycles=args.max_cycles,
    )

    print("\n" + "=" * 50)
    print("Evolution Summary")
    print("=" * 50)
    for report in reports:
        print(report.summary())
        print("-" * 30)
    print(f"Total cycles: {len(reports)}")
    if reports:
        print(f"Final score: {reports[-1].post_mean_score:.4f}")


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

    def _add_run_args(p: argparse.ArgumentParser, *, prompt_required: bool = True) -> None:
        p.add_argument(
            "--comfyui-addr",
            default=_server_addr(),
            metavar="HOST:PORT",
            help=(
                "ComfyUI server address, e.g. '127.0.0.1:7130'. "
                "Default: COMFYUI_ADDR env var or 127.0.0.1:8188"
            ),
        )
        p.add_argument(
            "--workflow",
            default=None,
            metavar="PATH",
            help="Path to API-format ComfyUI workflow JSON (omit to start from scratch)",
        )
        p.add_argument(
            "--prompt",
            required=prompt_required,
            default=None,
            help="Image generation prompt"
            + ("" if prompt_required else " (optional — comes from ComfyUI panel)"),
        )
        p.add_argument(
            "--model",
            default=_env_str("COMFYCLAW_MODEL", "anthropic/claude-sonnet-4-5"),
            metavar="MODEL",
            help=(
                "LiteLLM model string for the agent, e.g. 'anthropic/claude-sonnet-4-5', "
                "'openai/gpt-4o', 'gemini/gemini-2.0-flash', 'ollama/llama3.1'. "
                "Set the matching provider API key env-var."
            ),
        )
        p.add_argument(
            "--verifier-model",
            default=_env_str("COMFYCLAW_VERIFIER_MODEL", ""),
            metavar="MODEL",
            help=(
                "LiteLLM model string for the vision verifier (must support images). "
                "Defaults to the same value as --model. "
                "Example: --model ollama/llama3.1 --verifier-model openai/gpt-4o"
            ),
        )
        p.add_argument(
            "--iterations", type=int, default=_env_int("COMFYCLAW_MAX_ITERATIONS", 3), metavar="N"
        )
        p.add_argument(
            "--threshold",
            type=float,
            default=_env_float("COMFYCLAW_THRESHOLD", 0.85),
            metavar="SCORE",
        )
        p.add_argument(
            "--sync-port", type=int, default=_env_int("COMFYCLAW_SYNC_PORT", 8765), metavar="PORT"
        )
        p.add_argument("--no-sync", action="store_true", help="Disable live WebSocket sync")
        p.add_argument(
            "--skills-dir", default=os.environ.get("COMFYCLAW_SKILLS_DIR") or None, metavar="DIR"
        )
        p.add_argument(
            "--reset-each-iter",
            action="store_true",
            default=not _env_bool("COMFYCLAW_EVOLVE_FROM_BEST", True),
            help="Disable topology accumulation (reset to base each iteration)",
        )
        p.add_argument(
            "--max-repair-attempts",
            type=int,
            default=_env_int("COMFYCLAW_MAX_REPAIR_ATTEMPTS", 2),
            metavar="N",
            help="Max agent repair attempts when ComfyUI rejects a workflow (default 2)",
        )
        p.add_argument(
            "--output-dir", default=None, metavar="DIR", help="Directory for saved output images"
        )
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
        p.add_argument(
            "--verifier-mode",
            default=_env_str("COMFYCLAW_VERIFIER_MODE", "vlm"),
            choices=["vlm", "human", "hybrid"],
            metavar="MODE",
            help=(
                "Verification mode: 'vlm' (default) uses a vision LLM, "
                "'human' collects feedback via ComfyUI panel or terminal, "
                "'hybrid' runs VLM first then lets a human accept or override."
            ),
        )

    run_p = sub.add_parser("run", help="Run the full agent–generate–verify loop")
    _add_run_args(run_p)
    run_p.set_defaults(func=lambda a: _cmd_run(a, dry=False))

    dry_p = sub.add_parser("dry-run", help="Run agent only (no ComfyUI execution)")
    _add_run_args(dry_p)
    dry_p.set_defaults(func=lambda a: _cmd_run(a, dry=True))

    serve_p = sub.add_parser(
        "serve",
        help="Start persistent server — listen for generation triggers from ComfyUI",
    )
    _add_run_args(serve_p, prompt_required=False)
    serve_p.set_defaults(func=_cmd_serve)

    inst_p = sub.add_parser("install-node", help="Symlink ComfyClaw-Sync custom node into ComfyUI")
    inst_p.add_argument(
        "--comfyui-dir",
        default=None,
        metavar="DIR",
        help="ComfyUI installation directory (or set COMFYUI_DIR in .env)",
    )
    inst_p.set_defaults(func=_cmd_install_node)

    np_p = sub.add_parser("node-path", help="Print path to the bundled ComfyClaw-Sync plugin")
    np_p.set_defaults(func=_cmd_node_path)

    # ── benchmark subcommand ──────────────────────────────────────────
    bench_p = sub.add_parser(
        "benchmark",
        help="Run T2I evaluation benchmarks (GenEval2, CREA, OneIG-EN)",
    )
    bench_p.add_argument("--suite", required=True, choices=["geneval2", "crea", "oneig"])
    bench_p.add_argument("--name", required=True, help="Experiment name")
    bench_p.add_argument("--data-path", required=True, help="Path to benchmark JSONL data")
    bench_p.add_argument("--output-dir", default="benchmark_results")
    bench_p.add_argument("--max-iterations", type=int, default=3)
    bench_p.add_argument("--num-workers", type=int, default=1)
    bench_p.add_argument("--max-prompts", type=int, default=None)
    bench_p.add_argument("--stage-gated", action="store_true")
    bench_p.add_argument("--comfyui-addr", default=_server_addr())
    bench_p.add_argument("--model", default=_env_str("COMFYCLAW_MODEL", "anthropic/claude-sonnet-4-5"))
    bench_p.add_argument("--image-model", default=None)
    bench_p.add_argument("--workflow", default=None)
    bench_p.add_argument("--verifier-model", default=None)
    bench_p.add_argument("--verbose", action="store_true")
    bench_p.set_defaults(func=_cmd_benchmark)

    # ── evolve subcommand ─────────────────────────────────────────────
    evolve_p = sub.add_parser(
        "evolve",
        help="Run self-evolution cycle on benchmark results",
    )
    evolve_p.add_argument("--results", required=True, help="Path to benchmark results.json")
    evolve_p.add_argument("--evolved-skills-dir", default=None,
                          help="Directory for evolved skills (default: comfyclaw/skills_evolved/)")
    evolve_p.add_argument("--model", default=_env_str("COMFYCLAW_MODEL", "anthropic/claude-sonnet-4-5"))
    evolve_p.add_argument("--max-cycles", type=int, default=5)
    evolve_p.add_argument("--min-improvement", type=float, default=0.02)
    evolve_p.add_argument("--max-mutations", type=int, default=3)
    evolve_p.add_argument("--verbose", action="store_true")
    evolve_p.set_defaults(func=_cmd_evolve)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
