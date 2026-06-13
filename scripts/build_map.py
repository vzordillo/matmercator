#!/usr/bin/env python3
"""Thin wrapper for ``matmercator map``.

Kept for backwards compatibility. Install the package (``pip install -e .``)
and prefer ``matmercator map`` (or ``python -m matmercator map``); this shim
forwards its arguments to the ``map`` subcommand.
"""

from __future__ import annotations

import sys

from matmercator.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["map", *sys.argv[1:]]))
