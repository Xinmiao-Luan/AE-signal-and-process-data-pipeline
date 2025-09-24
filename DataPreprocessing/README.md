# DataPreprocessing

This folder contains preprocessing utilities that prepare **acoustic emission (AE) spectrograms** and **process logs** for downstream analysis (feature extraction, VAE training, clustering, SPC).

---

## Modules

### `dataClassification.py`
Implements process segmentation into **four classes** based on contour/infill transitions:

1. **Building the contour**  
   Rows with consecutive `"contour"` indicators.

2. **Transition: contour → infill**  
   Gap between two rows when the indicator switches from `"contour"` to `"infill"`.

3. **Building the infill**  
   Rows with consecutive `"infill"` indicators.

4. **Finalizing the cylinder**  
   Gap between two rows where the indicator transfers from `"infill"` or `"contour"`.  
   Represents nozzle repositioning or waiting for z-axis adjustment to begin a new layer.

---

### `dataTransfer.py`
- Reshapes spectrograms into a **fixed-size format**, enabling consistent input for feature extraction and model training.

### `frequency.py`
- Annotates the original spectrogram matrix with **real frequency information**, enriching the signal representation.

### `labelMatch.py`
- Aligns **spectrogram data** with **quality labels** based on time synchronization.  
- Ensures multimodal fusion between AE signals and external process data.

### `timeCalculator.py` & `timeTableProcess.py`
- Utilities for time alignment, process duration calculation, and structured storage of spectrograms.  
- According to classification, stores spectra **separately by operation stage**.

### `silentMomentDetector.py`
- Scans through AE spectrogram CSV files in a folder.  
- Flags files that contain **near-silent signals**.  

---

## Purpose
Together, these scripts form the **data preprocessing pipeline**:
- Segment raw process logs into meaningful phases  
- Reshape spectrograms  
- Enrich with frequency and label context  
- Save structured, aligned inputs for further ML/QA analysis  

This is the **first stage** of the pipeline before feature extraction, latent learning (VAE), and anomaly detection.
