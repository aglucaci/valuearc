"""Backward-compatible wrapper for the main screener entrypoint."""

from scripts.run_screener import main


if __name__ == "__main__":
    raise SystemExit(main())
