# Credit Scoring & Default Prediction

## Project Overview
This project focuses on building a robust quantitative model to predict borrower default using a large-scale credit dataset (149,563 observations). It simulates the model development process required for **IFRS 9** Probability of Default (PD) estimation.

## Key Methodologies
* **Data Engineering**: Conducted extensive data cleaning, addressing missing values and removing unrealistic records to ensure model robustness.
* **Class Imbalance Handling**: Implemented **SMOTE** and downsampling techniques to create a balanced 50/50 dataset, ensuring the model effectively learns minority (default) class patterns.
* **Machine Learning**: Developed and benchmarked multiple classifiers including Logistic Regression, Decision Tree, and **Random Forest**.
* **Performance**: Achieved a peak **AUC of 0.990** with the Random Forest model.
