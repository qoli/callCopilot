#!/usr/bin/python3
"""Copilot CLI BYOK launcher."""

from __future__ import annotations

import atexit
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_OMLX_MODEL = "Qwen3.6-35B-A3B-bf16"
DEEPSEEK_ALIAS = "ds"
DEEPSEEK_MODEL = "deepseek-v4-pro"
NVIDIA_DEEPSEEK_ALIAS = "nds"
NVIDIA_DEEPSEEK_MODEL = "deepseek-ai/deepseek-v4-pro"


PROXY_PROCESSES: List[Tuple[subprocess.Popen[Any], Path]] = []


def eprint(message: str) -> None:
    print(message, file=sys.stderr)


def usage() -> None:
    eprint(
        """Usage: callCopilot [model] [-- copilot-args...]
       callCopilot [copilot-args...]

Model aliases:
  ds    DeepSeek Anthropic API, model deepseek-v4-pro. Requires DEEPSEEK_API_KEY.
  nds   NVIDIA hosted DeepSeek OpenAI API, model deepseek-ai/deepseek-v4-pro. Requires NVIDIA_DEEPSEEK_API_KEY.

Default behavior: adds --autopilot unless already provided."""
    )


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        os.environ[key] = value


def load_config() -> None:
    env_file = os.environ.get("CALLCOPILOT_ENV_FILE")
    if env_file:
        load_dotenv(Path(env_file).expanduser())
    load_dotenv(ROOT / ".env")


def require_command(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise SystemExit(f"error: required command not found: {name}")
    return path


def parse_args(argv: List[str]) -> Tuple[str, List[str]]:
    if argv and argv[0] in {"-h", "--help"}:
        usage()
        raise SystemExit(0)

    requested = ""
    remaining = list(argv)
    if remaining and not remaining[0].startswith("-"):
        requested = remaining.pop(0)
    if remaining[:1] == ["--"]:
        remaining.pop(0)

    copilot_args = remaining or ["--autopilot"]
    if "--autopilot" not in copilot_args:
        copilot_args = ["--autopilot", *copilot_args]
    return requested, copilot_args


def resolve_model_id(requested: str) -> str:
    if requested == DEEPSEEK_ALIAS:
        return DEEPSEEK_MODEL
    if requested == NVIDIA_DEEPSEEK_ALIAS:
        return NVIDIA_DEEPSEEK_MODEL
    return requested or env("CALLCOPILOT_DEFAULT_MODEL", DEFAULT_OMLX_MODEL)


def env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value is not None and value != "" else default


def secret_arg_present(args: List[str]) -> bool:
    return any(arg == "--secret-env-vars" or arg.startswith("--secret-env-vars=") for arg in args)


def add_secret_env_args(args: List[str]) -> List[str]:
    if secret_arg_present(args):
        return args
    return [
        "--secret-env-vars=DEEPSEEK_API_KEY,NVIDIA_DEEPSEEK_API_KEY,COPILOT_PROVIDER_API_KEY,COPILOT_PROVIDER_BEARER_TOKEN",
        *args,
    ]


def fetch_json(url: str, token: str, method: str = "GET", body: Optional[bytes] = None) -> Any:
    request = Request(
        url,
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def validate_model_status(status: Dict[str, Any]) -> None:
    if not isinstance(status.get("loaded_count"), int) or not isinstance(status.get("models"), list):
        raise SystemExit("error: oMLX /v1/models/status returned unexpected JSON")
    for model in status["models"]:
        if not isinstance(model.get("id"), str) or not isinstance(model.get("loaded"), bool):
            raise SystemExit("error: oMLX model status entry is missing id/loaded")


def resolve_model_limits(status: Dict[str, Any], model_id: str) -> Tuple[str, str]:
    for model in status["models"]:
        if model.get("id") == model_id:
            prompt = model.get("max_context_window")
            output = model.get("max_tokens")
            if isinstance(prompt, int) and isinstance(output, int):
                return str(prompt), str(output)
            raise SystemExit(f"error: oMLX model has no token limits: {model_id}")
    raise SystemExit(f"error: target model not found in oMLX: {model_id}")


def should_skip_unload(status: Dict[str, Any], model_id: str) -> bool:
    loaded = [m.get("id") for m in status["models"] if m.get("loaded") is True]
    return status.get("loaded_count") == 1 and loaded == [model_id]


def unload_loaded_models(status: Dict[str, Any], server_root: str, token: str) -> None:
    loaded_any = False
    for model in status["models"]:
        if model.get("loaded") is not True:
            continue
        model_id = model["id"]
        loaded_any = True
        eprint(f"[callCopilot] unloading loaded model: {model_id}")
        fetch_json(f"{server_root}/v1/models/{quote(model_id, safe='')}/unload", token, method="POST", body=b"{}")
    if not loaded_any:
        eprint("[callCopilot] no loaded models detected; skipping unload")


def run_copilot(env_overrides: Dict[str, str], copilot_args: List[str]) -> int:
    require_command("copilot")
    child_env = os.environ.copy()
    child_env.update(env_overrides)
    return subprocess.call(["copilot", *copilot_args], env=child_env)


def ensure_thinking_fix_package() -> Path:
    package_version = env("CALLCOPILOT_DEEPSEEK_THINKING_FIX_VERSION", "0.2.5")
    prefix = Path(env("CALLCOPILOT_DEEPSEEK_THINKING_FIX_PREFIX", str(ROOT / ".runtime" / "deepseek-thinking-fix")))
    package_root = prefix / "node_modules" / "opencode-deepseek-thinking-fix"
    if (package_root / "dist" / "api.js").exists():
        return package_root

    require_command("npm")
    prefix.mkdir(parents=True, exist_ok=True)
    eprint(f"[callCopilot] installing opencode-deepseek-thinking-fix@{package_version}")
    subprocess.check_call(
        [
            "npm",
            "install",
            "--prefix",
            str(prefix),
            "--no-save",
            "--silent",
            f"opencode-deepseek-thinking-fix@{package_version}",
        ]
    )
    return package_root


def start_proxy(script: Path, env_vars: Dict[str, str], upstream_base_url: str) -> str:
    require_command("node")
    runtime_dir = Path(tempfile.mkdtemp(prefix="callcopilot-proxy-"))
    ready_file = runtime_dir / "ready"
    log_file = runtime_dir / "proxy.log"
    child_env = os.environ.copy()
    child_env.update(env_vars)

    log_handle = log_file.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            "node",
            str(script),
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--ready-file",
            str(ready_file),
            "--upstream-base-url",
            upstream_base_url,
        ],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=child_env,
    )
    log_handle.close()
    PROXY_PROCESSES.append((process, runtime_dir))

    for _ in range(200):
        if ready_file.exists() and ready_file.stat().st_size > 0:
            return ready_file.read_text(encoding="utf-8").strip()
        if process.poll() is not None:
            raise SystemExit(f"error: proxy exited early\n{read_file_best_effort(log_file)}")
        time.sleep(0.1)
    raise SystemExit(f"error: proxy did not become ready\n{read_file_best_effort(log_file)}")


def read_file_best_effort(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def cleanup_proxies() -> None:
    while PROXY_PROCESSES:
        process, runtime_dir = PROXY_PROCESSES.pop()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
        shutil.rmtree(str(runtime_dir), ignore_errors=True)


def install_signal_handlers() -> None:
    def handler(signum: int, _frame: Any) -> None:
        cleanup_proxies()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def run_omlx(model_id: str, copilot_args: List[str]) -> int:
    server_root = env("CALLCOPILOT_OMLX_SERVER_ROOT", "http://127.0.0.1:8001").rstrip("/")
    provider_base = env("CALLCOPILOT_OMLX_PROVIDER_BASE_URL", f"{server_root}/v1")
    token = env("CALLCOPILOT_OMLX_API_KEY", "")

    status = fetch_json(f"{server_root}/v1/models/status", token)
    validate_model_status(status)
    prompt_limit, output_limit = resolve_model_limits(status, model_id)
    if should_skip_unload(status, model_id):
        eprint(f"[callCopilot] target model already exclusively loaded: {model_id}")
    else:
        unload_loaded_models(status, server_root, token)

    proxy_base = start_proxy(
        ROOT / "callcopilot" / "proxies" / "openai_compat_cleanup_proxy.mjs",
        {"OPENAI_COMPAT_API_KEY": token},
        provider_base,
    )

    return run_copilot(
        {
            "COPILOT_PROVIDER_BASE_URL": f"{proxy_base}/v1",
            "COPILOT_PROVIDER_TYPE": "openai",
            "COPILOT_PROVIDER_BEARER_TOKEN": token,
            "COPILOT_PROVIDER_MAX_PROMPT_TOKENS": prompt_limit,
            "COPILOT_PROVIDER_MAX_OUTPUT_TOKENS": output_limit,
            "COPILOT_MODEL": model_id,
        },
        copilot_args,
    )


def run_deepseek(model_id: str, copilot_args: List[str]) -> int:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise SystemExit("error: DEEPSEEK_API_KEY is required for model alias 'ds'")

    package_root = ensure_thinking_fix_package()
    proxy_base = start_proxy(
        ROOT / "callcopilot" / "proxies" / "deepseek_anthropic_thinking_proxy.mjs",
        {
            "DEEPSEEK_API_KEY": api_key,
            "DEEPSEEK_THINKING_FIX_PACKAGE_ROOT": str(package_root),
        },
        env("CALLCOPILOT_DEEPSEEK_PROVIDER_BASE_URL", "https://api.deepseek.com/anthropic"),
    )

    return run_copilot(
        {
            "COPILOT_PROVIDER_BASE_URL": proxy_base,
            "COPILOT_PROVIDER_TYPE": "anthropic",
            "COPILOT_PROVIDER_API_KEY": api_key,
            "COPILOT_PROVIDER_MAX_PROMPT_TOKENS": env("CALLCOPILOT_DEEPSEEK_MAX_PROMPT_TOKENS", "1048576"),
            "COPILOT_PROVIDER_MAX_OUTPUT_TOKENS": env("CALLCOPILOT_DEEPSEEK_MAX_OUTPUT_TOKENS", "393216"),
            "COPILOT_MODEL": model_id,
        },
        add_secret_env_args(copilot_args),
    )


def run_nvidia_deepseek(model_id: str, copilot_args: List[str]) -> int:
    api_key = os.environ.get("NVIDIA_DEEPSEEK_API_KEY", "")
    if not api_key:
        raise SystemExit("error: NVIDIA_DEEPSEEK_API_KEY is required for model alias 'nds'")

    proxy_base = start_proxy(
        ROOT / "callcopilot" / "proxies" / "nvidia_deepseek_openai_proxy.mjs",
        {"NVIDIA_DEEPSEEK_API_KEY": api_key},
        env("CALLCOPILOT_NVIDIA_DEEPSEEK_PROVIDER_BASE_URL", "https://integrate.api.nvidia.com/v1"),
    )

    return run_copilot(
        {
            "COPILOT_PROVIDER_BASE_URL": f"{proxy_base}/v1",
            "COPILOT_PROVIDER_TYPE": "openai",
            "COPILOT_PROVIDER_API_KEY": api_key,
            "COPILOT_PROVIDER_MAX_PROMPT_TOKENS": env("CALLCOPILOT_NVIDIA_DEEPSEEK_MAX_PROMPT_TOKENS", "1048576"),
            "COPILOT_PROVIDER_MAX_OUTPUT_TOKENS": env("CALLCOPILOT_NVIDIA_DEEPSEEK_MAX_OUTPUT_TOKENS", "16384"),
            "COPILOT_MODEL": model_id,
        },
        add_secret_env_args(copilot_args),
    )


def main(argv: List[str]) -> int:
    atexit.register(cleanup_proxies)
    install_signal_handlers()
    load_config()
    requested, copilot_args = parse_args(argv)
    model_id = resolve_model_id(requested)

    if requested in {DEEPSEEK_ALIAS, DEEPSEEK_MODEL}:
        return run_deepseek(model_id, copilot_args)
    if requested in {NVIDIA_DEEPSEEK_ALIAS, NVIDIA_DEEPSEEK_MODEL}:
        return run_nvidia_deepseek(model_id, copilot_args)
    return run_omlx(model_id, copilot_args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
