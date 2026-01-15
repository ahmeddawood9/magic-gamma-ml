# MAGIC Gamma vs Hadron Classification

> **Project Status:** Active
> Pipeline now includes **Preprocessing**, **k-Nearest Neighbors (kNN)**, and **Naive Bayes** classification.

---

## Overview

This project implements a complete machine learning pipeline for the MAGIC Gamma Telescope dataset. The goal is to classify particle events as either **Gamma (signal)** or **Hadron (background noise)**.

The project features a modular architecture separating data preprocessing, visualization, and model implementation.

---

## Dataset

- **Name:** MAGIC Gamma Telescope Dataset
- **Source:** UCI Machine Learning Repository
- **Task:** Binary Classification
  - `gamma (1)` → signal
  - `hadron (0)` → background noise
- **Features:** 10 continuous features (fLength, fWidth, fSize, etc.)

---

## Project Pipeline

### 1. Data Loading & Cleaning
- Loads raw CSV data from `data/magic04.data`.
- Converts categorical labels (`g` / `h`) into numeric format (`1` / `0`).

### 2. Feature Visualization
- Generates probability density histograms for all 10 features.
- Visualizes the separation between Gamma and Hadron events.
- **Output:** Plots are saved in `outputs/plots/`.

### 3. Data Splitting
- **60% Training**: Used for model fitting.
- **20% Validation**: Used for tuning.
- **20% Testing**: Used for final evaluation.

### 4. Preprocessing (Scaling & Balancing)
- **Feature Scaling:** StandardScaler (mean = 0, std = 1) is applied to normalize feature ranges.
- **Class Balancing:** RandomOverSampler is applied **only to the training set** to handle class imbalance, ensuring models don't become biased toward the majority class.

### 5. Classification Models
The following models have been implemented and integrated into `main.py`:

* **k-Nearest Neighbors (kNN)**
    * Uses Euclidean distance.
    * Current setting: `k=5`.
    * Standard classifier for pattern recognition based on feature proximity.

* **Naive Bayes**
    * Gaussian Naive Bayes implementation.
    * Assumes feature independence; effective for high-dimensional data baselines.

---

## How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Pipeline:**
    Execute the main script from the project root:
    ```bash
    python -m src.main
    ```

---

## Outputs

* **Console:** Prints dataset shapes and classification reports (Precision, Recall, F1-Score) for both models.
* **Files:** Feature distribution plots saved in `outputs/plots/`.
