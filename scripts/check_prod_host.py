"""Preflight checks for a production Docker host.

Usage:
    python scripts/check_prod_host.py .env.docker.prod.local
    python scripts/check_prod_host.py .env.docker.prod.local --skip-dns
"""

from __future__ import annotations

import argparse
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.validate_prod_env import _load_env, validate  # noqa: E402


@dataclass(frozen=True)
class PreflightReport:
    errors: list[str]
    warnings: list[str]
    infos: list[str]


def _run_capture(command: list[str], *, cwd: Path) -> tuple[int, str]:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
    )
    output = (completed.stdout or completed.stderr or "").strip()
    return completed.returncode, output


def _check_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def _resolve_hostname(hostname: str) -> list[str]:
    addresses: list[str] = []
    for info in socket.getaddrinfo(hostname, None):
        ip = info[4][0]
        if ip not in addresses:
            addresses.append(ip)
    return addresses


def _free_disk_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def run_preflight(
    env_path: Path,
    *,
    env_data: dict[str, str] | None = None,
    validate_env: bool = True,
    skip_docker: bool = False,
    skip_ports: bool = False,
    skip_dns: bool = False,
) -> PreflightReport:
    errors: list[str] = []
    warnings: list[str] = []
    infos: list[str] = []

    env_data = env_data or _load_env(env_path)
    if validate_env:
        validate_errors, validate_warnings = validate(env_data)
        errors.extend(validate_errors)
        warnings.extend(validate_warnings)

    compose_path = ROOT_DIR / "docker-compose.prod.yml"
    caddyfile_path = ROOT_DIR / "deploy" / "Caddyfile"
    if not compose_path.exists():
        errors.append(f"Missing compose file: {compose_path}")
    else:
        infos.append(f"Found compose file: {compose_path}")
    if not caddyfile_path.exists():
        errors.append(f"Missing Caddyfile: {caddyfile_path}")
    else:
        infos.append(f"Found Caddyfile: {caddyfile_path}")

    markdown_docs = ROOT_DIR / "markdown_docs"
    runtime_dir = ROOT_DIR / "runtime"
    if not markdown_docs.exists():
        warnings.append(f"{markdown_docs} does not exist; API will start without local markdown documents.")
    else:
        infos.append(f"Found markdown docs directory: {markdown_docs}")
    if not runtime_dir.exists():
        warnings.append(f"{runtime_dir} does not exist yet; Docker will create it on first start.")
    else:
        infos.append(f"Found runtime directory: {runtime_dir}")

    free_disk_gb = _free_disk_gb(ROOT_DIR)
    if free_disk_gb < 5:
        warnings.append(f"Low disk space: only {free_disk_gb:.1f} GB free on the deployment volume.")
    else:
        infos.append(f"Disk free space looks OK: {free_disk_gb:.1f} GB")

    if not skip_ports:
        for port in (80, 443):
            if _check_port_available(port):
                infos.append(f"Port {port} is available.")
            else:
                errors.append(f"Port {port} is already in use; Caddy will not be able to bind it.")

    if not skip_dns:
        for key in ("APP_DOMAIN", "API_DOMAIN"):
            hostname = env_data.get(key, "").strip()
            if not hostname:
                continue
            try:
                resolved = _resolve_hostname(hostname)
            except OSError as exc:
                warnings.append(f"{key} ({hostname}) does not resolve yet: {exc}")
                continue
            if resolved:
                infos.append(f"{key} resolves to: {', '.join(resolved)}")
            else:
                warnings.append(f"{key} ({hostname}) did not return any DNS records.")

    if skip_docker:
        return PreflightReport(errors=errors, warnings=warnings, infos=infos)

    docker_path = shutil.which("docker")
    if not docker_path:
        errors.append("`docker` is not on PATH.")
        return PreflightReport(errors=errors, warnings=warnings, infos=infos)

    infos.append(f"Using docker binary: {docker_path}")

    for command, label in (
        (["docker", "--version"], "docker CLI"),
        (["docker", "compose", "version"], "docker compose"),
        (["docker", "info"], "docker daemon"),
        (
            ["docker", "compose", "--env-file", str(env_path), "-f", "docker-compose.prod.yml", "config", "--services"],
            "docker compose config",
        ),
    ):
        code, output = _run_capture(command, cwd=ROOT_DIR)
        if code != 0:
            errors.append(f"{label} check failed: {output or 'unknown error'}")
        else:
            first_line = output.splitlines()[0] if output else "ok"
            infos.append(f"{label} OK: {first_line}")

    return PreflightReport(errors=errors, warnings=warnings, infos=infos)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("env_file", nargs="?", default=".env.docker.prod.local")
    parser.add_argument("--skip-docker", action="store_true", help="Skip docker and compose command checks.")
    parser.add_argument("--skip-ports", action="store_true", help="Skip checks for host ports 80 and 443.")
    parser.add_argument("--skip-dns", action="store_true", help="Skip APP_DOMAIN/API_DOMAIN DNS resolution checks.")
    return parser.parse_args(argv[1:])


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    env_path = (ROOT_DIR / args.env_file).resolve() if not Path(args.env_file).is_absolute() else Path(args.env_file)
    if not env_path.exists():
        print(f"ERROR: {env_path} does not exist.")
        return 1

    report = run_preflight(
        env_path,
        skip_docker=args.skip_docker,
        skip_ports=args.skip_ports,
        skip_dns=args.skip_dns,
    )
    for info in report.infos:
        print(f"INFO: {info}")
    for warning in report.warnings:
        print(f"WARN: {warning}")
    for error in report.errors:
        print(f"ERROR: {error}")

    if report.errors:
        return 1

    print(f"OK: {env_path} passed production host preflight.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
