MAGIC Gamma Telescope Classification

This repository contains a modular machine learning pipeline designed to classify atmospheric Cherenkov gamma-ray telescope data. The project evaluates several classification strategies—from statistical baselines to deep learning—to identify high-energy gamma rays against background hadronic noise.
System Architecture

The project is structured with a "plug-in" architecture to ensure code reusability and clean separation of concerns.
1. Data Pipeline

    Preprocessing: The raw MAGIC dataset is cleaned and converted into a binary format (Gamma=1, Hadron=0).

    Scaling: We use standard scaling to ensure all 10 geometric features have a mean of 0 and unit variance. This is critical for distance-based models like KNN and SVM.

    Oversampling: Because the dataset is naturally imbalanced, the training set is balanced using oversampling. This prevents the models from developing a bias toward the majority class.

2. Modeling Strategy

I implemented a suite of models to compare different mathematical approaches to the classification problem:

    Logistic Regression: Used as the linear baseline to check for simple separability.

    Support Vector Machine (SVM): Implemented with an RBF kernel to handle non-linear boundaries. This currently yields the highest accuracy (~86%).

    Neural Network: A multi-layer perceptron (MLP) built with Keras. I implemented a Grid Search to tune hyperparameters (nodes, learning rate, batch size) and utilized Dropout and EarlyStopping to mitigate overfitting.

    k-Nearest Neighbors (KNN): A distance-based approach to observe how local clusters form in the feature space.

    Naive Bayes: A probabilistic baseline used to test the assumption of feature independence.
Results
Comparative Metrics
Model Architecture	           Accuracy	Macro F1-Score	Recall (Class 0)
Support Vector Machine (SVM)	0.86	0.84	0.79
Neural Network (MLP)	        0.85	0.83	0.82
k-Nearest Neighbors	            0.80	0.78	0.72
Logistic Regression	            0.78	0.76	0.71
Naive Bayes	                    0.72	0.65	0.39

Implementation Details

The code is optimized for Linux-native environments (tested on Arch Linux/Hyprland). To maintain stability and keep the system lightweight, the following configurations were applied:

    CPU Execution: Explicitly forced TensorFlow to use CPU to avoid CUDA driver overhead in non-GPU environments.

    Modularization: Each model and preprocessing step is isolated in the src/ directory to allow for independent testing.

How to Run
Prerequisites

    Python 3.10+

    Virtual environment tool (venv)

Installation

    Clone the repository: git clone https://github.com/yourusername/magic-gamma-ml.git

    Create and activate a virtual environment: python -m venv venv && source venv/bin/activate

    Install dependencies: pip install -r requirements.txt

Execution

Run the main orchestrator to execute the full pipeline: python -m main

The script will output classification reports for each model and save training history plots to the outputs/plots/ directory.
