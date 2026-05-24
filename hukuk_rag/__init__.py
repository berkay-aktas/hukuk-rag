"""Top-level convenience package — delegates to src/ for the real code.

This shim exists so ``python -m hukuk_rag <command>`` resolves to the CLI
without having to run ``python -m src.pipeline.cli``. Keeps the user-facing
entrypoint short.
"""
