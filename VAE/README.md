# VAE (Variational Autoencoder)

This folder contains implementations of **Variational Autoencoders (VAE)** and utilities for extracting, labeling, and visualizing latent representations of spectrogram features. These modules enable unsupervised learning of compact feature spaces from acoustic emission (AE) data.

---

## Modules

### `VariationalAutoencodersImp.py`
- Core **VAE implementation**.
- Input: spectrograms
- Defines encoder, latent reparameterization, and decoder networks.  
- Supports training, saving, and reconstruction for unsupervised feature extraction.
- Output: latent variable data (since we are exploring the features from the spectrogram here).

### `latentVariablesLabeledWithTimestamps.py`
- Aligns latent variables with **timestamps** and external **process data**.  
- Enables multimodal integration between AE spectrogram features and sensor/process logs.

### `latentVariablesProcessing.py`
- Handles **processing of latent variables** produced by the VAE.  
- Includes utilities for t-SNE for further dimensionality reduction.
- Includes some data Visualization utilities.

### `plot.py`
- Store some other Visualization utilities for latent spaces and reconstructions.  

---

## Purpose
The VAE module provides the **representation learning backbone** of the pipeline:

1. Train a VAE on AE spectrogram features.  
2. Extract latent variables as compact feature vectors.  
3. Align them with process sensor data for multimodal analysis.  
4. Visualize the learned latent space for interpretability and downstream clustering/SPC.

This forms the **unsupervised learning stage** of the overall AE signal + process data pipeline.

