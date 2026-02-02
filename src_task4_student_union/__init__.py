#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 4 package for Student Union robust optimization.
"""

from __future__ import annotations

import os
import sys


def ensure_src_on_path() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(repo_root, "src")
    if src_dir not in sys.path:
        sys.path.append(src_dir)

