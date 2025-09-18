# SpectrumProcessing

This folder contains utilities for **spectrogram processing** and **time alignment** of acoustic emission (AE) signals. These scripts are used to visualize, annotate, and synchronize spectrograms with external process data, preparing them for feature extraction and downstream ML analysis.

---

## Modules

### `drawspectrum.py`
- Generates standard spectrogram visualizations from AE signals.  
- Provides a clean view of frequency vs. time to analyze energy distributions.

### `drawspectrumWithMarkers.py`
- Extends basic spectrogram plotting by overlaying **markers** that highlight key events or labeled timestamps.  
- Useful for inspecting how defects or process events appear in the frequency domain.

### `extractTimestampsFromSelectedLabels.py`
- Extracts **time intervals** from user-selected labels (e.g., process phases or quality indicators).  
- Supports aligning spectrogram data with process logs for multimodal fusion.

### `time_align.py`
- Synchronizes AE spectrograms with **external sensor/process timelines**.  
- Ensures consistent alignment between signal data and machine process data.

---

## Purpose
The scripts in this folder provide the **visualization and alignment layer** of the pipeline:
1. Generate and inspect spectrograms.  
2. Overlay labels/markers for better interpretability.  
3. Align signal data with process metadata (timestamps, operations).  
4. Output ready-to-use data for feature extraction and modeling.

This makes the spectrograms interpretable and usable alongside **process data and downstream ML workflows**.

