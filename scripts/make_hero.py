#!/usr/bin/env python3
"""Thin wrapper for ``matmercator hero``.

Kept for backwards compatibility. Install the package (``pip install -e .``)
and prefer ``matmercator hero`` (or ``python -m matmercator hero``); this shim
forwards its arguments to the ``hero`` subcommand.
"""

from __future__ import annotations

import sys

from matmercator.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["hero", *sys.argv[1:]]))
