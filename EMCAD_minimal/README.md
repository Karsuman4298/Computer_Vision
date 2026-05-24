# EMCAD Minimal Project

A compact EMCAD inference workspace for Synapse segmentation, including the `inference.ipynb` notebook and minimal model utilities.

## Overview

This folder contains a lightweight version of the EMCAD segmentation pipeline:

- `pvtv2.py`: Minimal pyramid vision transformer backbone implementation
- `decoders.py`: Efficient multi-scale convolutional attention decoding blocks
- `networks.py`: `EMCADNet` wrapper that combines encoder and decoder
- `demo.py`: Minimal forward-pass example
- `inference.ipynb`: Notebook for loading a trained checkpoint, running inference, visualizing results, and evaluating Dice score

## Notebook Usage

To run the inference notebook:

1. Open `EMCAD_minimal/inference.ipynb` in Jupyter or VS Code.
2. Change the checkpoint path to your local model file, for example:
   - `/Users/sumankar/Downloads/emcad_best.pth`
3. Change the dataset path if needed:
   - `./data/synapse/synapse/train_npz_new`
4. Execute notebook cells sequentially.

The notebook includes:

- model definition and device setup
- checkpoint loading and inference timing
- sample visualization with ground truth, prediction, overlay, and error map
- Dice score calculation for multi-organ segmentation

## Environment

Recommended environment setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you need GPU support, install a matching `torch` build for your CUDA version.

## Expected Files

- `sample.npz`: demo input and ground truth sample for inference
- `EMCAD_minimal/inference.ipynb`: main notebook to run and visualize predictions
- model checkpoint file: `emcad_best.pth` or `emcad_model.pth`

## Notes

- The notebook assumes grayscale 2D slices with shape `[H, W]`.
- If your checkpoint is a wrapper dict with `model_state_dict`, update the loading cell accordingly.
- Keep `EMCAD_minimal/README.md` and `inference.ipynb` together when pushing to GitHub.
