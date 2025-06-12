# Makemore Name Generator

This is a character-level neural network that generates realistic names using PyTorch.  
Inspired by Karpathy's "Zero to Hero" series.

## Features
- MLP model with embedding and 2 hidden layers
- Temperature-controlled name generation
- CLI interface for generating names
- Structured cleanly: `train.py`, `generate.py`, `model.py`, `vocab.py`

## How to Use

```bash
python train.py             # Train the model
python generate.py          # Generate 10 names (default)
python generate.py --n 20   # Generate 20 names
