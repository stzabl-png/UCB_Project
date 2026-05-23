#!/usr/bin/env python3
"""Entry point: human-prior supervision affordance training (L1 only, no FC)."""

import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

from model.affordance.train_hp import main

if __name__ == "__main__":
    main()
