import argparse
import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path
import importlib.resources as pkg_resources

STACK_FILES = ["docker-compose.yml", "Dockerfile", "schema.sql"]
CACHE_DIR = Path(os.path.expanduser("~/.cache/vectorgraph/stack"))


def _print_err(msg: str) -> None:
    sys.stderr.write(msg + "\n")


def ensure_docker() -> str:
    docker_bin = shutil.which("docker")
    if not docker_bin:
        _print_err("Docker is not installed or not on PATH. Install Docker Desktop: https://docs.docker.com/get-docker/")
        sys.exit(1)
    try:
        subprocess.run([docker_bin, "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError:
        _print_err("Docker is installed but not running. Start Docker Desktop and retry.")
        sys.exit(1)
    return docker_bin


def ensure_compose(docker_bin: str) -> list[str]:
    # Prefer `docker compose`
    try:
        subprocess.run([docker_bin, "compose", "version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return [docker_bin, "compose"]
    except Exception:
        pass
    compose_bin = shutil.which("docker-compose")
    if compose_bin:
        return [compose_bin]
    _print_err("Docker Compose not available. Install Compose: https://docs.docker.com/compose/install/")
    sys.exit(1)


def ensure_stack_files() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for fname in STACK_FILES:
        target = CACHE_DIR / fname
        if target.exists():
            continue
        with pkg_resources.files("vectorgraph.stack").joinpath(fname).open("rb") as src:
            target.write_bytes(src.read())
    return CACHE_DIR


def ensure_env_file(stack_dir: Path) -> Path | None:
    # Prefer current working directory .env
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        target = stack_dir / ".env"
        if not target.exists() or cwd_env.stat().st_mtime > target.stat().st_mtime:
            target.write_bytes(cwd_env.read_bytes())
        return target
    # Fall back to packaged .env.example if available
    packaged = pkg_resources.files("vectorgraph.stack").joinpath(".env.example")
    if packaged.exists():
        target = stack_dir / ".env"
        if not target.exists():
            target.write_bytes(packaged.read_bytes())
        return target
    return None


def resolve_env_file(stack_dir: Path) -> Path | None:
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        return cwd_env
    stack_env = stack_dir / ".env"
    if stack_env.exists():
        return stack_env
    return None


def run_compose(compose_cmd: list[str], stack_dir: Path, args: list[str], env_file: Path | None) -> int:
    cmd = compose_cmd + ["-f", str(stack_dir / "docker-compose.yml")]
    if env_file:
        cmd += ["--env-file", str(env_file)]
    cmd += args
    try:
        result = subprocess.run(cmd, cwd=stack_dir)
        return result.returncode
    except FileNotFoundError:
        _print_err("Failed to run docker compose. Ensure Docker is installed.")
        return 1


def cmd_up(compose_cmd: list[str], stack_dir: Path, env_file: Path | None) -> int:
    rc = run_compose(compose_cmd, stack_dir, ["up", "-d"], env_file)
    if rc != 0:
        _print_err("docker compose up failed; check Docker is running and images are accessible.")
    return rc


def cmd_down(compose_cmd: list[str], stack_dir: Path, env_file: Path | None) -> int:
    return run_compose(compose_cmd, stack_dir, ["down"], env_file)


def cmd_logs(compose_cmd: list[str], stack_dir: Path, env_file: Path | None, follow: bool) -> int:
    args = ["logs"] + (["-f"] if follow else [])
    return run_compose(compose_cmd, stack_dir, args, env_file)


def cmd_ps(compose_cmd: list[str], stack_dir: Path, env_file: Path | None) -> int:
    return run_compose(compose_cmd, stack_dir, ["ps"], env_file)


def copy_example() -> int:
    target = Path.cwd() / "demo.py"
    if target.exists():
        _print_err(f"{target} already exists; not overwriting.")
        return 1
    try:
        with pkg_resources.files("vectorgraph.examples").joinpath("demo.py").open("rb") as src:
            target.write_bytes(src.read())
        print(f"Wrote example to {target}")
        return 0
    except FileNotFoundError:
        _print_err("Example file not found in package.")
        return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vectorgraph", description="Manage the VectorGraph Docker stack")
    sub = p.add_subparsers(dest="command", required=True)

    up = sub.add_parser("up", help="Start the stack (Docker + Postgres + embeddings)")
    up.set_defaults(func="up")

    down = sub.add_parser("down", help="Stop the stack")
    down.set_defaults(func="down")

    logs = sub.add_parser("logs", help="Tail logs")
    logs.add_argument("--follow", "-f", action="store_true", help="Follow log output")
    logs.set_defaults(func="logs")

    ps = sub.add_parser("ps", help="List services")
    ps.set_defaults(func="ps")

    demo = sub.add_parser("demo", help="Copy demo example into the current directory")
    demo.set_defaults(func="demo")

    mcp = sub.add_parser("mcp", help="Run the MCP stdio server")
    mcp.set_defaults(func="mcp")

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # MCP server does not require Docker; run and exit.
    if args.func == "mcp":
        from vectorgraph.mcp_server import main as mcp_main

        asyncio.run(mcp_main())
        return

    docker_bin = ensure_docker()
    compose_cmd = ensure_compose(docker_bin)
    stack_dir = ensure_stack_files()
    env_file = ensure_env_file(stack_dir)

    cmd_map = {
        "up": cmd_up,
        "down": cmd_down,
        "logs": lambda c, s, e: cmd_logs(c, s, e, args.follow),
        "ps": cmd_ps,
        "demo": lambda *_: copy_example(),
    }
    handler = cmd_map.get(args.func)
    rc = handler(compose_cmd, stack_dir, env_file)
    sys.exit(rc)


if __name__ == "__main__":
    main()
