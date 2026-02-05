# ARENA Workshop: Transformers & Mechanistic Interpretability

A comprehensive workshop covering transformer architecture implementation and mechanistic interpretability analysis using PyTorch and transformer-lens.

## Overview

This workshop consists of two notebooks that build your understanding from first principles:

1. **Transformers from Scratch** – Implement a complete transformer architecture in PyTorch without external libraries
2. **Introduction to Mechanistic Interpretability** – Learn to analyze transformer internals using transformer-lens

## Workshop Materials

### Notebook 1: Transformers from Scratch

Build a transformer model from the ground up to understand each component (embeddings, attention, MLPs, layer normalization, etc.).

- **Exercises**: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DurhamAISafety/5-feb-ARENA-workshop/blob/main/1.1_Transformer_from_Scratch_exercises.ipynb)
- **Solutions**: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DurhamAISafety/5-feb-ARENA-workshop/blob/main/1.1_Transformer_from_Scratch_solutions.ipynb)

### Notebook 2: Intro to Mechanistic Interpretability

Learn to dissect and understand how transformers process information internally using the transformer-lens library.

- **Exercises**: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DurhamAISafety/5-feb-ARENA-workshop/blob/main/1.2_Intro_to_Mech_Interp_exercises.ipynb)
- **Solutions**: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DurhamAISafety/5-feb-ARENA-workshop/blob/main/1.2_Intro_to_Mech_Interp_solutions.ipynb)

**Supplementary**: [ARENA Streamlit Page](https://arena-chapter1-transformer-interp.streamlit.app/02_[1.2]_Intro_to_Mech_Interp)

## Setup

### Using Google Colab (Recommended)

The easiest way to run these notebooks is via Google Colab (links above). No setup required.

### Local Setup

For local development, set up your environment with `uv`:

```bash
# Install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

Alternatively, using traditional methods:

```bash
pip install -e .
jupyter notebook
```

## Prerequisites

- Basic Python knowledge
- Familiarity with PyTorch fundamentals
- Understanding of neural networks basics

## Requirements

- Python 3.11+
- PyTorch
- transformer-lens
- Jupyter notebook support

See `pyproject.toml` for the complete dependency list.