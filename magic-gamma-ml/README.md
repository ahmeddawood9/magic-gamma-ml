# MAGIC Gamma vs Hadron Classification – Preprocessing Pipeline

> **Note:** This repository currently contains only the preprocessing and data preparation pipeline.  
> Machine learning models (KNN, SVM, Neural Networks, etc.) will be added later.

---

## Overview

This project demonstrates a clean, scalable machine learning pipeline using the MAGIC Gamma Telescope dataset.  
The focus at this stage is **data preprocessing**, which is critical for building accurate and reliable ML models.

The current pipeline handles:
- Loading raw data
- Cleaning and converting labels
- Visualizing feature distributions
- Splitting dataset into train, validation, and test sets
- Scaling features
- Handling class imbalance (training set only)

---

## Dataset

- **Name:** MAGIC Gamma Telescope Dataset  
- **Source:** UCI Machine Learning Repository  
- **Task:** Classify particle events as:
  - `gamma (1)` → signal
  - `hadron (0)` → background noise

The dataset has **10 continuous features** and **1 class label**.

---

## Current Pipeline (Preprocessing Stage)

### 1. Data Loading
- Load raw CSV data from `data/magic04.data`
- Assign descriptive column names
- Convert string class labels (`g` / `h`) into numeric (`1` / `0`)

### 2. Feature Visualization
- Generate histograms for each feature
- Compare gamma vs hadron distributions
- Plots are saved in `outputs/plots/`  
  *(Useful for exploratory data analysis and understanding separability)*

### 3. Train / Validation / Test Split
- Dataset is **shuffled** to avoid bias
- Split proportions:
  - 60% Training
  - 20% Validation
  - 20% Test
- Ensures fair evaluation later

### 4. Feature Scaling
- StandardScaler is applied to make features **mean = 0, std = 1**
- Prevents features with large numeric ranges from dominating learning

### 5. Handling Class Imbalance
- RandomOverSampler applied **only on training data**
- Ensures training set is balanced while validation and test sets remain realistic

---

## Current Outputs

### Plots
Saved in `outputs/plots/`:
