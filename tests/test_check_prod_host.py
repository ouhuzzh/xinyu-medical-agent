import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.check_prod_host import run_preflight  # noqa: E402


VALID_ENV = """\
APP_DOMAIN=medical.prod.internal
API_DOMAIN=api.medical.prod.internal
PUBLIC_API_BASE_URL=https://api.medical.prod.internal
API_CORS_ORIGINS=https://medical.prod.internal
APP_ENV=production
JWT_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
CHECKPOINT_SIGNING_KEY=yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy
API_AUTH_TOKENS_JSON={}
POSTGRES_DB=ai_companion
POSTGRES_USER=postgres
POSTGRES_PASSWORD=zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
ACTIVE_LLM_PROVIDER=deepseek
ACTIVE_EMBEDDING_PROVIDER=openai_compatible
LLM_MODEL=demo-llm
EMBEDDING_MODEL=demo-embedding
VECTOR_DIMENSION=1024
DEEPSEEK_API_KEY=
OPENAI_API_KEY=
INSTALL_LOCAL_ML=false
MCP_ENABLED=false
USER_MEMORY_ENCRYPT_PII=true
MCP_TOKEN_ENCRYPTION_KEYS=9qNwD0nSS3MqQb5d9fD3YFPoY7_nN0slVdAy8W6kvVE=
"""


class CheckProdHostTests(unittest.TestCase):
    def _write_env(self) -> Path:
        handle = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".env")
        with handle:
            handle.write(VALID_ENV)
        self.addCleanup(lambda: os.unlink(handle.name) if os.path.exists(handle.name) else None)
        return Path(handle.name)

    @mock.patch("scripts.check_prod_host._run_capture")
    @mock.patch("scripts.check_prod_host._resolve_hostname")
    @mock.patch("scripts.check_prod_host._check_port_available")
    @mock.patch("scripts.check_prod_host._free_disk_gb", return_value=42.0)
    @mock.patch("scripts.check_prod_host.shutil.which", return_value="/usr/bin/docker")
    def test_run_preflight_reports_success(
        self,
        _mock_which,
        _mock_disk,
        mock_ports,
        mock_resolve,
        mock_run,
    ):
        env_path = self._write_env()
        mock_ports.return_value = True
        mock_resolve.side_effect = [["203.0.113.10"], ["203.0.113.11"]]
        mock_run.side_effect = [
            (0, "Docker version 28.0.0"),
            (0, "Docker Compose version v2.0.0"),
            (0, "Server: Docker Engine"),
            (0, "caddy\npostgres\nredis\napi\nfrontend"),
        ]

        report = run_preflight(env_path)

        self.assertEqual(report.errors, [])
        self.assertIn("Port 80 is available.", report.infos)
        self.assertTrue(any("APP_DOMAIN resolves to" in item for item in report.infos))

    @mock.patch("scripts.check_prod_host._run_capture")
    @mock.patch("scripts.check_prod_host._check_port_available", return_value=False)
    @mock.patch("scripts.check_prod_host._free_disk_gb", return_value=3.0)
    @mock.patch("scripts.check_prod_host.shutil.which", return_value="/usr/bin/docker")
    def test_run_preflight_reports_port_conflict_and_docker_failures(
        self,
        _mock_which,
        _mock_disk,
        _mock_port,
        mock_run,
    ):
        env_path = self._write_env()
        mock_run.side_effect = [
            (1, "docker unavailable"),
            (0, "Docker Compose version v2.0.0"),
            (1, "daemon down"),
            (0, "api"),
        ]

        report = run_preflight(env_path, skip_dns=True)

        self.assertTrue(any("Port 80 is already in use" in item for item in report.errors))
        self.assertTrue(any("docker CLI check failed" in item for item in report.errors))
        self.assertTrue(any("Low disk space" in item for item in report.warnings))


if __name__ == "__main__":
    unittest.main()
