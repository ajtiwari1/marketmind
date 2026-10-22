import os
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

from libs.langfuse import flush_langfuse, get_langfuse_client, observe, propagate_attributes


class LangfusePackageTests(unittest.TestCase):
    def test_import_does_not_construct_client_without_credentials(self) -> None:
        env = os.environ.copy()
        for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
            env.pop(name, None)

        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import langfuse\n"
                    "from unittest.mock import patch\n"
                    "with patch.object(langfuse, 'get_client') as factory:\n"
                    "    import libs.langfuse\n"
                    "    factory.assert_not_called()\n"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )

    def test_public_helpers_are_importable(self) -> None:
        self.assertTrue(callable(get_langfuse_client))
        self.assertTrue(callable(flush_langfuse))
        self.assertTrue(callable(observe))
        self.assertTrue(callable(propagate_attributes))

    @patch("libs.langfuse.client.get_client")
    def test_get_langfuse_client_returns_sdk_client(self, sdk_get_client: Mock) -> None:
        expected_client = Mock()
        sdk_get_client.return_value = expected_client

        actual_client = get_langfuse_client()

        self.assertIs(actual_client, expected_client)
        sdk_get_client.assert_called_once_with()

    @patch("libs.langfuse.client.get_langfuse_client")
    def test_flush_langfuse_flushes_sdk_client(self, get_client: Mock) -> None:
        client = Mock()
        get_client.return_value = client

        flush_langfuse()

        client.flush.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
