import os
import unittest
from unittest.mock import Mock, patch

from examples.langfuse_trace_demo import run_demo


class LangfuseDemoTests(unittest.TestCase):
    @patch("examples.langfuse_trace_demo.get_client")
    def test_demo_requires_credentials_before_creating_client(
        self, get_client: Mock
    ) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "LANGFUSE_PUBLIC_KEY"):
                run_demo()

        get_client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
