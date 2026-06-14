import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project"))

from core.runtime_security import collect_security_issues_from_settings  # noqa: E402


class RuntimeSecurityTests(unittest.TestCase):
    def test_development_allows_default_jwt_with_warning(self):
        errors, warnings = collect_security_issues_from_settings(
            {
                "APP_ENV": "development",
                "JWT_SECRET_KEY": "change-me-in-production-please",
                "CHECKPOINT_SIGNING_KEY": "",
                "MCP_TOKEN_ENCRYPTION_KEYS": "",
                "MCP_TOKEN_ENCRYPTION_KEY": "",
                "USER_MEMORY_ENABLED": True,
                "USER_MEMORY_ENCRYPT_PII": True,
                "MCP_ENABLED": True,
            }
        )

        self.assertEqual(errors, [])
        self.assertTrue(any("default development value" in item for item in warnings))

    def test_production_requires_checkpoint_and_encryption_keys(self):
        errors, _warnings = collect_security_issues_from_settings(
            {
                "APP_ENV": "production",
                "JWT_SECRET_KEY": "x" * 48,
                "CHECKPOINT_SIGNING_KEY": "",
                "MCP_TOKEN_ENCRYPTION_KEYS": "",
                "MCP_TOKEN_ENCRYPTION_KEY": "",
                "USER_MEMORY_ENABLED": True,
                "USER_MEMORY_ENCRYPT_PII": True,
                "MCP_ENABLED": False,
            }
        )

        self.assertTrue(any("CHECKPOINT_SIGNING_KEY" in item for item in errors))
        self.assertTrue(any("MCP_TOKEN_ENCRYPTION_KEY" in item for item in errors))

    def test_production_rejects_unencrypted_user_memory(self):
        errors, _warnings = collect_security_issues_from_settings(
            {
                "APP_ENV": "production",
                "JWT_SECRET_KEY": "x" * 48,
                "CHECKPOINT_SIGNING_KEY": "y" * 48,
                "MCP_TOKEN_ENCRYPTION_KEYS": "9qNwD0nSS3MqQb5d9fD3YFPoY7_nN0slVdAy8W6kvVE=",
                "USER_MEMORY_ENABLED": True,
                "USER_MEMORY_ENCRYPT_PII": False,
                "MCP_ENABLED": False,
            }
        )

        self.assertTrue(any("USER_MEMORY_ENCRYPT_PII=true" in item for item in errors))

    def test_production_accepts_valid_settings(self):
        errors, warnings = collect_security_issues_from_settings(
            {
                "APP_ENV": "production",
                "JWT_SECRET_KEY": "x" * 48,
                "CHECKPOINT_SIGNING_KEY": "y" * 48,
                "MCP_TOKEN_ENCRYPTION_KEYS": "9qNwD0nSS3MqQb5d9fD3YFPoY7_nN0slVdAy8W6kvVE=",
                "USER_MEMORY_ENABLED": True,
                "USER_MEMORY_ENCRYPT_PII": True,
                "MCP_ENABLED": True,
            }
        )

        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
