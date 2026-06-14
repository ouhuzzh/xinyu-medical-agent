import os
import sys
import unittest

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.validate_prod_env import validate  # noqa: E402


class ValidateProdEnvTests(unittest.TestCase):
    def test_validate_rejects_unencrypted_user_memory_in_production(self):
        errors, warnings = validate(
            {
                "APP_DOMAIN": "medical.example.com",
                "API_DOMAIN": "api.medical.example.com",
                "PUBLIC_API_BASE_URL": "https://api.medical.example.com",
                "API_CORS_ORIGINS": "https://medical.example.com",
                "APP_ENV": "production",
                "JWT_SECRET_KEY": "x" * 48,
                "CHECKPOINT_SIGNING_KEY": "y" * 48,
                "POSTGRES_DB": "ai_companion",
                "POSTGRES_USER": "postgres",
                "POSTGRES_PASSWORD": "z" * 32,
                "ACTIVE_LLM_PROVIDER": "deepseek",
                "ACTIVE_EMBEDDING_PROVIDER": "openai_compatible",
                "LLM_MODEL": "demo-llm",
                "EMBEDDING_MODEL": "demo-embedding",
                "VECTOR_DIMENSION": "1024",
                "MCP_TOKEN_ENCRYPTION_KEYS": "9qNwD0nSS3MqQb5d9fD3YFPoY7_nN0slVdAy8W6kvVE=",
                "USER_MEMORY_ENCRYPT_PII": "false",
                "USER_MEMORY_ENABLED": "true",
                "MCP_ENABLED": "false",
                "API_AUTH_TOKENS_JSON": "{}",
            }
        )

        self.assertEqual(warnings, [])
        self.assertTrue(any("USER_MEMORY_ENCRYPT_PII=true" in item for item in errors))


if __name__ == "__main__":
    unittest.main()
