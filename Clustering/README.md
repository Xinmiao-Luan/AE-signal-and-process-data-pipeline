# Clustering (Latent Variable Analysis)

This folder contains methods for analyzing **latent variables** extracted from spectrograms and process data. The focus is on **dimensionality reduction** and **unsupervised clustering**, which are key steps in exploring the structure of the learned feature space.

---

## Modules

### `Cluster.py`
- Provides general clustering utilities for latent features.  
- Can be extended to support multiple algorithms (KMeans, DBSCAN, etc.).  
- Outputs cluster assignments for downstream analysis.

### `PCA.py`
- Applies **Principal Component Analysis (PCA)** to latent variables or feature vectors.  
- Used for dimensionality reduction, visualization, and noise filtering.  
- Helps interpret how features vary across process phases.

### `reconstructionErrorsSelection.py`
- Evaluates reconstruction errors from the **VAE** or other models.  
- Uses reconstruction error as a measure of anomaly detection or sample quality.  
- Supports thresholding and selection of abnormal segments.

---

## Purpose
The Clustering module enables:
1. **Dimensionality reduction** (PCA) for visualization and noise removal.  
2. **Clustering of latent features** to identify structure in AE/process data.  
3. **Error-based anomaly detection** using reconstruction error analysis.  

This stage complements the **VAE representation learning** by grouping and validating latent spaces, providing insights into process quality and potential defects.
