# Clustering

This folder contains **unsupervised clustering methods** applied to latent variables with/without aligned process data. These methods help reveal structure in the data and flag potential process anomalies.

---

## Modules

### `DBSCAN.py`
- Implements **Density-Based Spatial Clustering of Applications with Noise (DBSCAN)**.  
- Groups points into clusters based on density.  

### `Distance.py`
- Provides **distance metric utilities** (e.g., Euclidean, cosine).  
- Used to compare latent vectors, cluster centroids, or AE features.  

### `Isolation Forest.py`
- Implements **Isolation Forest** anomaly detection.  
- Isolates rare points (defects or unusual events) with fewer splits.  

### `KMeans.py`
- Implements **KMeans clustering**.  
- Partitions latent vectors into `k` groups.  

---

## Purpose

The Clustering module enables:
1. **Grouping** of latent variables into process-related patterns.  
2. **Detection** of abnormal events via distance measures or isolation forests.  
3. **Comparison** of different clustering algorithms for robustness.  
