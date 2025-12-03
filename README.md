# Multi-Classifier-Project-O-B-C-Classifiers-

🧠 Multi-Classifier Project (O, B, C Classifiers)

This project contains three custom classifiers that I built from scratch, each handling a different dataset and applying a different classification method. The goal is to classify data points into specific label groups using rule-based logic, thresholding, and machine learning.

📌 Classifiers Overview
1️⃣ O-Classifier (Rule-Based)

Dataset: dataset_o7.csv

Uses 8 input features

Predicts one of 10 classes (o0–o9)

Classification is based on nested logical conditions

Designed and implemented manually through domain rules

2️⃣ B-Classifier (Threshold-Based)

Dataset: dataset_b7.csv

Binary classes: bn and bo

Histogram analysis showed a clear separation

Rule:

value < 0.5 → bn

value ≥ 0.5 → bo

3️⃣ C-Classifier (Logistic Regression Model)

Dataset: Multiple CSVs combined from data/ folder

Model: Logistic Regression

Trained to classify values into corresponding C-labels

Model saved using pickle and used for inference

🧩 Key Skills Demonstrated

Rule-based model design

Threshold classification

Logistic regression training

Dataset preprocessing and merging

Python (pandas, numpy, sklearn)

✔ Summary

This project shows my ability to build different types of classifiers—logical, statistical, and machine-learning based—and integrate them into a clean and functional system.
