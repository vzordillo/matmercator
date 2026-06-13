#!/usr/bin/env python3
"""Thin wrapper for ``matmercator features``.

Kept for backwards compatibility. Install the package (``pip install -e .``)
and prefer ``matmercator features`` (or ``python -m matmercator features``);
this shim forwards its arguments to the ``features`` subcommand.
"""

from __future__ import annotations

import sys

from matmercator.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["features", *sys.argv[1:]]))
