# Plan: reusable Langfuse utilities

## Goal

Create reusable Langfuse helpers under `libs/` for this project and future AI
applications. The target is a Langfuse server/platform v4, which is compatible
with Python SDK v5.

## Proposed structure

```text
libs/
  langfuse/
    __init__.py
    client.py
    tracing.py
    utils/
      __init__.py
      context.py
      decorators.py
```

## Steps

1. **Confirm the package setup** — complete
   - Use `pyproject.toml` to make the library installable.
   - Depend on Langfuse Python SDK v5 for compatibility with server v4.

2. **Create the package** — complete
   - Add `libs/langfuse` and `libs/langfuse/utils` as Python packages.
   - Set up client configuration and tracing helpers in separate modules.
   - Keep reusable utility functions under `utils/`.

3. **Make configuration reusable** — complete
   - Use the SDK's standard environment variables for credentials and host
     configuration.
   - Avoid hard-coded secrets and app-specific configuration.
   - Make it possible for consuming applications to configure or provide a
     Langfuse client.

4. **Define the public imports** — complete
   - Export the intended reusable helpers from package `__init__.py` files.
   - Keep implementation details internal unless they are needed by consumers.

5. **Document usage** — complete
   - Document required environment variables and the Langfuse v4 dependency.
   - Show how to import and use the package from this project and another
     application.

6. **Add focused tests** — complete
   - Test configuration and helper behavior.
   - Confirm importing the package does not require credentials or connect to
     Langfuse immediately.

## Implementation note

The repository uses `pyproject.toml` with setuptools package discovery for
`libs.*`. Consumers can install the repository (for example, with
`pip install -e .`) and import helpers from `libs.langfuse`.
