#!/usr/bin/env python3
"""Thin wrapper for ``matmercator landscapes``.

Kept for backwards compatibility. Install the package (``pip install -e .``)
and prefer ``matmercator landscapes`` (or ``python -m matmercator landscapes``);
this shim forwards its arguments to the ``landscapes`` subcommand.
"""

from __future__ import annotations

import sys

from matmercator.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["landscapes", *sys.argv[1:]]))
