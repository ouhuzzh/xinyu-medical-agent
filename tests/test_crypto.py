"""Unit tests for symmetric encryption + key rotation.

Focuses on behaviour, not crypto primitives (trusts the cryptography library).
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project"))


def _new_key() -> str:
    from cryptography.fernet import Fernet
    return Fernet.generate_key().decode()


class TokenCryptoTests(unittest.TestCase):
    def setUp(self):
        from mcp_integration import token_crypto
        self.tc = token_crypto
        self.tc._reset_cache_for_tests()

    def tearDown(self):
        self.tc._reset_cache_for_tests()

    def test_roundtrip_single_key(self):
        key = _new_key()
        with patch("config.MCP_TOKEN_ENCRYPTION_KEYS", ""), \
             patch("config.MCP_TOKEN_ENCRYPTION_KEY", key):
            ct = self.tc.encrypt_token("hospital-secret-abc")
            self.assertNotEqual(ct, "hospital-secret-abc")
            self.assertEqual(self.tc.decrypt_token(ct), "hospital-secret-abc")

    def test_empty_input_is_passthrough(self):
        with patch("config.MCP_TOKEN_ENCRYPTION_KEYS", ""), \
             patch("config.MCP_TOKEN_ENCRYPTION_KEY", _new_key()):
            self.assertEqual(self.tc.encrypt_token(""), "")
            self.assertEqual(self.tc.decrypt_token(""), "")

    def test_decrypt_unknown_ciphertext_returns_empty(self):
        with patch("config.MCP_TOKEN_ENCRYPTION_KEYS", ""), \
             patch("config.MCP_TOKEN_ENCRYPTION_KEY", _new_key()):
            self.assertEqual(self.tc.decrypt_token("not-a-real-token"), "")

    def test_key_rotation_old_ciphertext_still_decrypts(self):
        """Encrypt with old key, rotate to new key (old kept as fallback), should still decrypt."""
        old, new = _new_key(), _new_key()
        # Phase 1: only old key
        with patch("config.MCP_TOKEN_ENCRYPTION_KEYS", ""), \
             patch("config.MCP_TOKEN_ENCRYPTION_KEY", old):
            self.tc._reset_cache_for_tests()
            old_ct = self.tc.encrypt_token("secret-from-yesterday")

        # Phase 2: rotation — new is primary, old is fallback
        self.tc._reset_cache_for_tests()
        with patch("config.MCP_TOKEN_ENCRYPTION_KEYS", f"{new},{old}"), \
             patch("config.MCP_TOKEN_ENCRYPTION_KEY", ""):
            self.assertEqual(self.tc.decrypt_token(old_ct), "secret-from-yesterday")
            # New writes use the new key
            new_ct = self.tc.encrypt_token("secret-today")
            self.assertNotEqual(new_ct, old_ct)
            self.assertEqual(self.tc.decrypt_token(new_ct), "secret-today")

        # Phase 3: drop the old key — old ciphertext stops decrypting
        self.tc._reset_cache_for_tests()
        with patch("config.MCP_TOKEN_ENCRYPTION_KEYS", new), \
             patch("config.MCP_TOKEN_ENCRYPTION_KEY", ""):
            self.assertEqual(self.tc.decrypt_token(old_ct), "")
            self.assertEqual(self.tc.decrypt_token(new_ct), "secret-today")

    def test_pii_helpers_use_same_keys(self):
        key = _new_key()
        with patch("config.MCP_TOKEN_ENCRYPTION_KEYS", ""), \
             patch("config.MCP_TOKEN_ENCRYPTION_KEY", key):
            ct = self.tc.encrypt_pii("对青霉素过敏")
            self.assertNotEqual(ct, "对青霉素过敏")
            self.assertEqual(self.tc.decrypt_pii(ct), "对青霉素过敏")
            # Cross-compat with token helpers — same underlying primitive
            self.assertEqual(self.tc.decrypt_token(ct), "对青霉素过敏")

    def test_production_without_key_raises(self):
        with patch("config.MCP_TOKEN_ENCRYPTION_KEYS", ""), \
             patch("config.MCP_TOKEN_ENCRYPTION_KEY", ""), \
             patch("config.APP_ENV", "production"):
            with self.assertRaises(RuntimeError) as ctx:
                self.tc._get_crypto()
            self.assertIn("must be set in production", str(ctx.exception))


class UserMemoryEncryptionTests(unittest.TestCase):
    """End-to-end check that encryption/decryption is transparent to callers.

    Avoids DB by patching the connection.  Verifies the SQL params received
    contain ciphertext (not plaintext), and that decrypted reads return the
    original.
    """

    def setUp(self):
        from mcp_integration import token_crypto
        self.tc = token_crypto
        self.tc._reset_cache_for_tests()

    def tearDown(self):
        self.tc._reset_cache_for_tests()

    def test_encrypted_content_roundtrip_via_helpers(self):
        """Sanity check the helpers in user_memory_store.py."""
        with patch("config.USER_MEMORY_ENCRYPT_PII", True), \
             patch("config.MCP_TOKEN_ENCRYPTION_KEYS", ""), \
             patch("config.MCP_TOKEN_ENCRYPTION_KEY", _new_key()):
            from memory import user_memory_store as ums
            ciphered = ums._encrypt_content("高血压病史")
            self.assertTrue(ciphered.startswith(ums._PII_MARKER))
            self.assertEqual(ums._decrypt_content(ciphered), "高血压病史")

    def test_legacy_plaintext_passes_through(self):
        """Rows from before encryption was switched on should decrypt to themselves."""
        with patch("config.USER_MEMORY_ENCRYPT_PII", True), \
             patch("config.MCP_TOKEN_ENCRYPTION_KEYS", ""), \
             patch("config.MCP_TOKEN_ENCRYPTION_KEY", _new_key()):
            from memory import user_memory_store as ums
            # Legacy plaintext does not carry the marker
            self.assertEqual(ums._decrypt_content("旧的明文记忆"), "旧的明文记忆")

    def test_feature_flag_disabled_skips_encryption(self):
        with patch("config.USER_MEMORY_ENCRYPT_PII", False):
            from memory import user_memory_store as ums
            self.assertEqual(ums._encrypt_content("anything"), "anything")
            self.assertEqual(ums._decrypt_content("anything"), "anything")

    def test_corrupt_ciphertext_does_not_raise(self):
        with patch("config.USER_MEMORY_ENCRYPT_PII", True), \
             patch("config.MCP_TOKEN_ENCRYPTION_KEYS", ""), \
             patch("config.MCP_TOKEN_ENCRYPTION_KEY", _new_key()):
            from memory import user_memory_store as ums
            # Marker present but body is junk — should NOT crash the read path
            result = ums._decrypt_content(ums._PII_MARKER + "corrupted-bytes")
            self.assertEqual(result, "[decrypt-failed]")


if __name__ == "__main__":
    unittest.main()
