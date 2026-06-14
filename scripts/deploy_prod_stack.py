"""Validate, start, and smoke-test the production Docker stack.

Usage:
    python scripts/deploy_prod_stack.py .env.docker.prod.local
    python scripts/deploy_prod_stack.py .env.docker.prod.local --skip-smoke
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.check_prod_host import run_preflight  # noqa: E402
from scripts.smoke_docker_deploy import main as smoke_main  # noqa: E402
from scripts.validate_prod_env import _load_env, validate  # noqa: E402


def _run(command: list[str], *, cwd: Path) -> int:
    print("+", " ".join(command))
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    return completed.returncode


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("env_file", nargs="?", default=".env.docker.prod.local")
    parser.add_argument("--skip-build", action="store_true", help="Skip docker compose --build.")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip post-deploy smoke checks.")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip production host preflight checks.")
    parser.add_argument("--skip-dns", action="store_true", help="Skip APP_DOMAIN/API_DOMAIN DNS checks during preflight.")
    parser.add_argument("--skip-ports", action="store_true", help="Skip host port 80/443 checks during preflight.")
    return parser.parse_args(argv[1:])


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    env_path = (ROOT_DIR / args.env_file).resolve() if not Path(args.env_file).is_absolute() else Path(args.env_file)
    if not env_path.exists():
        print(f"ERROR: {env_path} does not exist.")
        return 1

    env_data = _load_env(env_path)
    errors, warnings = validate(env_data)
    for warning in warnings:
        print(f"WARN: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        return 1

    if not args.skip_preflight:
        report = run_preflight(
            env_path,
            env_data=env_data,
            validate_env=False,
            skip_dns=args.skip_dns,
            skip_ports=args.skip_ports,
        )
        for info in report.infos:
            print(f"INFO: {info}")
        for warning in report.warnings:
            print(f"WARN: {warning}")
        for error in report.errors:
            print(f"ERROR: {error}")
        if report.errors:
            return 1

    compose_cmd = [
        "docker",
        "compose",
        "--env-file",
        str(env_path),
        "-f",
        "docker-compose.prod.yml",
        "up",
        "-d",
    ]
    if not args.skip_build:
        compose_cmd.insert(-1, "--build")
    rc = _run(compose_cmd, cwd=ROOT_DIR)
    if rc != 0:
        return rc

    if args.skip_smoke:
        print("OK: production stack started; smoke test skipped.")
        return 0

    app_domain = env_data.get("APP_DOMAIN", "").strip()
    api_base_url = env_data.get("PUBLIC_API_BASE_URL", "").strip()
    if app_domain:
        os.environ["FRONTEND_URL"] = f"https://{app_domain}"
    if api_base_url:
        os.environ["API_BASE_URL"] = api_base_url
    if env_data.get("API_AUTH_TOKENS_JSON", "").strip() not in {"", "{}"}:
        print("INFO: set API_AUTH_TOKEN manually if you want /api/system/status included in smoke output.")

    return smoke_main()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
