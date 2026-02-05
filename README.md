# 5 Feb'26 ARENA Workshop: Transformers & Mechanistic Interpretability

​This week we'll be hosting a workshop based on ARENA's mechanistic interpretability course. We'll focus on explaining and using transformer_lens - a popular mech interp python library. ​Sneak peek at the code here - we'll see how far we get through the first 2 sections!

Lots of:
- Python
- PyTorch
- TransformerLens (1.2 notebook)

NOTE - these notebooks are directly copied from ARENA, except the setup cells have been changed because the old original ones broke.

[ARENA Streamlit Page](https://arena-chapter1-transformer-interp.streamlit.app/)

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

## Setup

### Using Google Colab (Recommended)

The easiest way to run these notebooks is via Google Colab (links above). No setup required.

#### Saving Your Work in Colab

To persist your changes and keep your work safe, you have two options:

**Option 1: Save to Google Drive (Recommended)**
1. Open the notebook in Colab using the links above
2. Click **File** → **Save a copy in Drive**
3. This creates a copy in your Google Drive that you can edit and save normally
4. All changes are automatically saved to your Drive

**Option 2: Download the Notebook**
1. Click **File** → **Download** → **Download .ipynb**
2. Save the notebook locally on your computer
3. You can later upload it back to Colab to continue working

> **Tip**: If you don't save a copy to Drive, your changes will be lost when you close the tab. Always save to Drive if you plan to revisit your work!

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