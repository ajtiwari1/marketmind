"""Send a sample application trace and Python log records to Langfuse."""

import logging
import os
import sys
from uuid import uuid4

from langfuse import get_client

from libs.langfuse import propagate_attributes, trace_logs


logger = logging.getLogger("marketmind.langfuse_demo")
logger.setLevel(logging.INFO)
logger.propagate = False


@trace_logs(logger_name=logger.name, name="langfuse-log-demo")
def send_sample_trace() -> None:
    """Create a sample trace and log messages within its observation."""
    client = get_client()
    session_id = f"langfuse-demo-{uuid4()}"

    with propagate_attributes(
        user_id="langfuse-demo-user",
        session_id=session_id,
        tags=["langfuse-demo"],
    ):
        logger.info("Demo request started")

        with client.start_as_current_observation(
            as_type="generation",
            name="sample-generation",
            model="demo-model",
            input={"question": "What does this demo send to Langfuse?"},
        ) as generation:
            logger.info("Running the simulated model step")
            answer = (
                "The trace contains a decorated function observation, a sample "
                "generation, and Python log messages as child spans."
            )
            generation.update(output={"answer": answer})

        logger.info("Demo request completed")


def run_demo() -> None:
    """Check credentials, run the decorated example, and flush its trace."""
    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        raise RuntimeError(
            "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env before "
            "running this example."
        )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(console_handler)
    try:
        send_sample_trace()
    finally:
        try:
            get_client().flush()
        finally:
            logger.removeHandler(console_handler)
            console_handler.close()


if __name__ == "__main__":
    try:
        run_demo()
    except RuntimeError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
