# marketmind

## Reusable Langfuse utilities

The Langfuse helpers live in `libs.langfuse` and use the official Python SDK.
For a Langfuse server/platform v4, this project depends on Python SDK v5.

Install [uv](https://docs.astral.sh/uv/) and Python 3.11 or newer, then sync
the project dependencies and create the virtual environment and lockfile:

```bash
uv sync
```

Add your Langfuse project credentials to the root `.env` file. Keep this file
local and never commit it. For Langfuse Cloud, the SDK's default host can be
used; set `LANGFUSE_HOST` in `.env` only when using a custom/self-hosted host:

```dotenv
LANGFUSE_PUBLIC_KEY=your-public-key
LANGFUSE_SECRET_KEY=your-secret-key
# Optional:
# LANGFUSE_HOST=https://your-langfuse-host
```

Run commands with the `.env` file loaded by uv:

```bash
uv run --env-file .env python your_app.py
```

To send a real example trace, generation, and Python log messages to Langfuse:

```bash
uv run --env-file .env python examples/langfuse_trace_demo.py
```

The demo creates a request trace with a simulated generation. Its dedicated
logger writes to the console and records each demo log message as a child span.
Do not send secrets or sensitive data in trace inputs or log messages.

Use the helpers in an application:

```python
from libs.langfuse import (
    flush_langfuse,
    propagate_attributes,
    trace_logs,
)
import logging

logger = logging.getLogger("my_app.worker")

@trace_logs(logger_name="my_app.worker", name="answer-question")
def answer_question(question: str) -> str:
    logger.info("Starting answer generation")
    return "..."

with propagate_attributes(user_id="user-123"):
    answer_question("What is Langfuse?")

# Flush buffered events during application shutdown.
flush_langfuse()
```

`@trace_logs` wraps the function in a Langfuse span and records messages from
the named logger (and its child loggers) as child spans for that call. Use
`@observe` directly when you want function tracing without log capture.

The SDK manages client construction and configuration from environment
variables. Importing the helpers does not create a client; call
`get_langfuse_client()` when the application needs it. Use
`uv run --env-file .env python -m unittest discover -s tests -v` to run the
tests.