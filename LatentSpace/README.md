# LatentSpace

This folder contains utilities for **analyzing and interpreting latent variables** learned from spectrograms and process data. The focus is on **dimensionality reduction**, **clustering**, and **error-based anomaly detection**.

---

## Modules

### `Cluster.py`
- Provides clustering utilities for latent features.  
- Can be extended to support algorithms such as KMeans, DBSCAN, etc.  

### `PCA.py`
- Applies **Principal Component Analysis (PCA)** to latent variables or AE features.  
- Reduces dimensionality for visualization and noise reduction.  

### `reconstructionErrorsSelection.py`
- Collects and marks high reconstruction errors from the **VAE** models.  
- Identifies samples with poor reconstructions, which may indicate defects or unusual behavior.

---

## Purpose

The LatentSpace module enables:
1. **Exploration** of latent variables through clustering and PCA.  
2. **Visualization** of high-dimensional embeddings in 2D/3D.  
3. **Evaluation** of quality via reconstruction error analysis.  
