# AE Signal and Process Data Pipeline

This repository implements a modular pipeline for **acoustic emission (AE) signals** and **process data** in additive manufacturing processes.  
It includes preprocessing, spectrogram processing, feature extraction (VAE), latent space analysis, clustering, and anomaly detection.  

---

## Folder Documentation

### DataPreprocessing
Preprocessing utilities to prepare **AE spectrograms** and **process logs** for downstream analysis.

**Modules**
- `dataClassification.py`  
  Implements segmentation into four classes:  
  1. **Building the contour** – rows with consecutive `"contour"` indicators  
  2. **Transition: contour → infill** – gap where indicator switches from `"contour"` to `"infill"`  
  3. **Building the infill** – rows with consecutive `"infill"` indicators  
  4. **Finalizing the cylinder** – gap where indicator transfers from `"infill"` or `"contour"`, nozzle repositioning or z-axis adjustment  

- `dataTransfer.py` → Reshapes spectrograms into fixed-size format for consistent input.  
- `frequency.py` → Annotates spectrogram matrix with real frequency information.  
- `labelMatch.py` → Aligns spectrogram data with quality labels via time synchronization.  
- `timeCalculator.py` & `timeTableProcess.py` → Time alignment, duration calculation, and structured storage.  

**Purpose**
- Segment raw logs into phases  
- Reshape spectrograms  
- Add frequency/label context  
- Save aligned inputs for ML/QA analysis  

---

### 📁 SpectrumProcessing
Utilities for **spectrogram processing and alignment** of AE signals with external process data.

**Modules**
- `drawspectrum.py` → Generates standard spectrogram visualizations.  
- `drawspectrumWithMarkers.py` → Adds markers highlighting labeled events.  
- `extractTimestampsFromSelectedLabels.py` → Extracts time intervals from user-selected labels.  
- `time_align.py` → Synchronizes AE spectrograms with process timelines.  

**Purpose**
- Generate and inspect spectrograms  
- Overlay markers for interpretability  
- Align signals with process metadata  
- Prepare data for modeling  

---

### VAE (Variational Autoencoder)
Implements **VAE training and feature extraction** for unsupervised learning from spectrograms.

**Modules**
- `VariationalAutoencodersImp.py` → Core VAE (encoder, latent reparam, decoder).  
  - **Input**: spectrograms  
  - **Output**: latent variable data  
- `latentVariablesLabeledWithTimestamps.py` → Aligns latent variables with timestamps and process logs.  
- `latentVariablesProcessing.py` → Post-processes latent variables; includes t-SNE + visualization.  
- `plot.py` → Additional visualization utilities for latent spaces/reconstructions.  

**Purpose**
- Train VAEs on spectrograms  
- Extract latent features  
- Align latent variables with sensor data  
- Visualize latent space for interpretability  

---

### LatentSpace
Utilities for **analyzing and interpreting latent variables**.

**Modules**
- `Cluster.py` → Clustering utilities (extendable to KMeans, DBSCAN, etc.).  
- `PCA.py` → Principal Component Analysis for dimensionality reduction.  
- `reconstructionErrorsSelection.py` → Identifies and flags high reconstruction errors.  

**Purpose**
- Explore latent variables with PCA/clustering  
- Visualize embeddings in 2D/3D  
- Detect anomalies via reconstruction error analysis  

---

### Clustering
Implements **unsupervised clustering + anomaly detection** on latent features.

**Modules**
- `DBSCAN.py` → Density-based clustering + outlier detection.  
- `Distance.py` → Distance metric utilities (Euclidean, cosine).  
- `Isolation Forest.py` → Isolation Forest anomaly detection.  
- `KMeans.py` → KMeans clustering baseline.  

**Purpose**
- Group latent variables into patterns  
- Detect abnormal events via distances or isolation forests  
- Compare clustering methods for robustness  

---

## Pipeline Flow
1. DataPreprocessing
    └── Clean, segment, and align AE signals with process logs
          ↓
2. SpectrumProcessing
    └── Generate spectrograms, add markers, align with process metadata
          ↓
3. VAE
    └── Train Variational Autoencoder and extract latent variables
          ↓
4.1 LatentSpace
    └── Dimensionality reduction (PCA), clustering utilities, error analysis

4.2 Clustering
    └── Apply DBSCAN, KMeans, Isolation Forest, and distance-based methods

