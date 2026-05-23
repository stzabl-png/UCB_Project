#!/usr/bin/env python3
"""
Entry point for affordance training (delegates to model.affordance.train).

用法:
    python model/train_affordance.py
    python model/train_affordance.py --gpus 0
    python -m model.affordance.train --gpus 0
"""
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

from model.affordance.train import main

if __name__ == "__main__":
    main()
