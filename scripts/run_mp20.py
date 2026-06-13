#!/usr/bin/env python3
"""Thin wrapper for ``matmercator run``.

Kept for backwards compatibility. Install the package (``pip install -e .``)
and prefer the ``matmercator`` command (or ``python -m matmercator``); this
shim just forwards its arguments to the ``run`` subcommand.
"""

from __future__ import annotations

import sys

from matmercator.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["run", *sys.argv[1:]]))
