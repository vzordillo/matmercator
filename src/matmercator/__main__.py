"""Enable ``python -m matmercator`` as an alias for the ``matmercator`` CLI."""

from matmercator.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
