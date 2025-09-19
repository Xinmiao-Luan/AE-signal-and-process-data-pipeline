# Clustering

This folder contains **unsupervised clustering and anomaly detection methods** applied to latent variables or spectrogram features. These methods help reveal structure in the data and flag potential process anomalies.

---

## Modules

### `DBSCAN.py`
- Implements **Density-Based Spatial Clustering of Applications with Noise (DBSCAN)**.  
- Groups points into clusters based on density.  
- Identifies noise/outliers that do not belong to any cluster.  
- Useful for detecting irregular spectrogram segments or process deviations.

### `Distance.py`
- Provides **distance metric utilities** (e.g., Euclidean, cosine).  
- Used to compare latent vectors, cluster centroids, or AE features.  
- Forms the basis for anomaly scoring and cluster evaluation.

### `Isolation Forest.py`
- Implements **Isolation Forest** anomaly detection.  
- Isolates rare points (defects or unusual events) with fewer splits.  
- Efficient for high-dimensional AE latent features.

### `KMeans.py`
- Implements **KMeans clustering**.  
- Partitions latent vectors into `k` groups.  
- Provides a simple and scalable baseline for unsupervised grouping.

---

## Purpose

The Clustering module enables:
1. **Grouping** of latent variables into process-related patterns.  
2. **Detection** of abnormal events via distance measures or isolation forests.  
3. **Comparison** of different clustering algorithms for robustness.  

This stage complements the **VAE latent space** by extracting meaningful structure and highlighting anomalies in AE + process data.
