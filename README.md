# DSCI 552 – Homework 7: SVMs for Multi-Label Classification and K-Means Clustering

Classifying frog calls by **family, genus, and species** from 22 MFCC audio features, using the UCI [Anuran Calls (MFCCs)](https://archive.ics.uci.edu/ml/datasets/Anuran+Calls+%28MFCCs%29) data set (7,195 calls).

## Contents

- Multi-label evaluation: exact match ratio, Hamming loss, and Hamming score
- Binary relevance with Gaussian-kernel SVMs (one-vs-all), tuned by 10-fold cross-validation, on raw and standardized features
- L1-penalized linear SVMs, with and without SMOTE inside the cross-validation folds
- K-means clustering with k chosen by the Calinski–Harabasz index, repeated 50 times (Monte Carlo)
- ISLR Exercise 12.6.2 (hierarchical clustering dendrograms)

## Results summary (test set)

| Model | Exact match | Hamming loss |
|---|---|---|
| Gaussian SVM (raw) | **0.982** | **0.011** |
| Gaussian SVM (standardized) | 0.981 | 0.013 |
| L1 linear SVM | 0.914 | 0.057 |
| L1 linear SVM + SMOTE | 0.872 | 0.067 |

Unsupervised k-means (k = 2 selected in all 50 runs) reaches a Hamming loss of 0.299 with majority-label assignment.

## How to run

1. Download and unzip the Anuran Calls data inside the repo folder (the data is not included).
2. Install the dependencies: `pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn scipy`
3. Open `HW7.ipynb` and run all cells. The SVM grid searches and the k-means Monte Carlo take several minutes each.
