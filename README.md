# Final Term Machine Learning and Deep Learning

## Project Overview

This repository contains the final term project for the Machine Learning and Deep Learning course. The project focuses on building end-to-end machine learning and deep learning pipelines using the provided datasets.

There are two main tasks in this repository:

1. **Fraud Detection Classification**

   * Predict whether an online transaction is fraudulent or not.
   * Target variable: `isFraud`.

2. **Song Release Year Regression**

   * Predict the release year of a song based on numerical audio features.
   * Target variable: the first column in the dataset.

Both projects include data preprocessing, feature engineering, model training, hyperparameter tuning using Optuna, model evaluation, interpretation, and experiment tracking using MLflow.

---

## Repository Purpose

The purpose of this repository is to demonstrate practical implementation of end-to-end machine learning and deep learning workflows. The workflow covers data loading, data understanding, preprocessing, model development, model evaluation, hyperparameter tuning, MLflow tracking, and result interpretation.

---

## Project 1: Fraud Detection Classification

### Objective

The objective of this project is to build an end-to-end classification model that can predict the probability of an online transaction being fraudulent.

### Dataset

The dataset used in this project is:

* `train_transaction.csv`

The dataset contains transaction-related features such as transaction amount, product code, card information, address information, and the target label `isFraud`.

The identity dataset is not used because `train_identity.csv` was not available in the provided dataset. Therefore, the classification pipeline was built using `train_transaction.csv` only.

### Workflow

The classification workflow includes:

1. Data loading
2. Data understanding
3. Missing value analysis
4. Dropping columns with too many missing values
5. Handling missing values
6. Encoding categorical features
7. Feature scaling
8. Train-test split
9. Handling class imbalance using class weight
10. Deep learning model training
11. Hyperparameter tuning using Optuna
12. Model evaluation
13. MLflow experiment tracking
14. Saving model and evaluation reports

### Models Used

The models implemented in this project are:

| Model                                    | Description                                                     |
| ---------------------------------------- | --------------------------------------------------------------- |
| Baseline MLP                             | Basic neural network model used as the first benchmark          |
| MLP with Dropout and Batch Normalization | Improved neural network architecture to reduce overfitting      |
| Optuna Tuned MLP                         | Deep learning model with hyperparameters optimized using Optuna |

### Evaluation Metrics

Because fraud detection is an imbalanced classification problem, the model is evaluated using several metrics:

| Metric           | Description                                                                        |
| ---------------- | ---------------------------------------------------------------------------------- |
| Accuracy         | Measures overall prediction correctness                                            |
| Precision        | Measures how many predicted fraud cases are actually fraud                         |
| Recall           | Measures how many actual fraud cases are detected by the model                     |
| F1-Score         | Harmonic mean of precision and recall                                              |
| ROC-AUC          | Measures the model's ability to distinguish between fraud and non-fraud            |
| PR-AUC           | Useful metric for imbalanced classification problems                               |
| Confusion Matrix | Shows true positive, true negative, false positive, and false negative predictions |

### Classification Result Summary

| Model                 |  Accuracy | Precision |    Recall |  F1-Score |   ROC-AUC |    PR-AUC |
| --------------------- | --------: | --------: | --------: | --------: | --------: | --------: |
| Baseline MLP          | isi nilai | isi nilai | isi nilai | isi nilai | isi nilai | isi nilai |
| MLP Dropout BatchNorm | isi nilai | isi nilai | isi nilai | isi nilai | isi nilai | isi nilai |
| Optuna Tuned MLP      | isi nilai | isi nilai | isi nilai | isi nilai | isi nilai | isi nilai |

### Classification Interpretation

For fraud detection, accuracy alone is not enough because the dataset is highly imbalanced. Recall is important because the model should be able to detect as many fraudulent transactions as possible. Precision is also important to reduce false fraud alerts. ROC-AUC and PR-AUC are used to evaluate the model more fairly in imbalanced classification settings.

---

## Project 2: Song Release Year Regression

### Objective

The objective of this project is to build an end-to-end regression pipeline that predicts the release year of a song based on numerical audio features.

### Dataset

The dataset used in this project is:

* `midterm-regresi-dataset.csv`

The first value in each row represents the target label, which is the song release year. The remaining values are numerical features extracted from the audio signal. Since the features do not have human-readable names, they are renamed as `feature_1`, `feature_2`, `feature_3`, and so on.

### Workflow

The regression workflow includes:

1. Data loading
2. Renaming columns
3. Data understanding
4. Missing value checking
5. Converting data into numerical format
6. Handling missing values using median
7. Handling target outliers
8. Handling feature outliers using IQR capping
9. Feature and target separation
10. Train-test split
11. Feature scaling
12. Regression model training
13. Hyperparameter tuning using Optuna
14. Regression model evaluation
15. Model interpretation using LIME
16. MLflow experiment tracking
17. Saving model, scaler, and reports

### Models Used

The models implemented in this project are:

| Model                                      | Description                                  |
| ------------------------------------------ | -------------------------------------------- |
| Linear Regression                          | Baseline regression model                    |
| Ridge Regression / Random Forest Regressor | Machine learning regression comparison model |
| Deep Learning MLP                          | Neural network regression model              |
| Optuna Tuned Deep Learning MLP             | Deep learning model optimized using Optuna   |

### Evaluation Metrics

The regression models are evaluated using the following metrics:

| Metric   | Description                                                      |
| -------- | ---------------------------------------------------------------- |
| MSE      | Mean Squared Error, measures average squared prediction error    |
| RMSE     | Root Mean Squared Error, shows error in the original target unit |
| MAE      | Mean Absolute Error, measures average absolute prediction error  |
| R² Score | Shows how well the model explains target variation               |

### Regression Result Summary

| Model                                      |       MSE |      RMSE |       MAE |        R² |
| ------------------------------------------ | --------: | --------: | --------: | --------: |
| Linear Regression                          | isi nilai | isi nilai | isi nilai | isi nilai |
| Ridge Regression / Random Forest Regressor | isi nilai | isi nilai | isi nilai | isi nilai |
| Deep Learning MLP                          | isi nilai | isi nilai | isi nilai | isi nilai |
| Optuna Tuned Deep Learning MLP             | isi nilai | isi nilai | isi nilai | isi nilai |

### Regression Interpretation

The best regression model is selected based on the lowest RMSE and MAE, as well as the highest R² score. RMSE and MAE show the prediction error in terms of years, while R² shows how well the model explains the variation in the release year.

LIME is used to interpret the prediction of the best regression model. It helps explain which audio features have the strongest influence on a specific prediction made by the model.

---

## MLflow Tracking

MLflow is used to track machine learning experiments. The tracked components include:

* Model parameters
* Hyperparameters from Optuna
* Evaluation metrics
* Model artifacts
* Evaluation reports
* Visualization results

In Google Colab, MLflow tracking is configured using SQLite as the backend store to avoid file store issues.

---

## Repository Structure

```text
finalterm-machine-learning/
│
├── notebooks/
│   ├── fraud_detection_classification.ipynb
│   └── song_year_regression.ipynb
│
├── reports/
│   ├── model_comparison.csv
│   ├── roc_curve.png
│   └── precision_recall_curve.png
│
├── reports_regression/
│   ├── regression_model_comparison.csv
│   ├── training_loss.png
│   ├── actual_vs_predicted.png
│   ├── error_distribution.png
│   └── lime_explanation.html
│
├── models/
│   └── best_fraud_detection_model.keras
│
├── models_regression/
│   ├── best_regression_model.keras
│   └── scaler.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/username/finalterm-machine-learning.git
cd finalterm-machine-learning
```

### 2. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset

Place the datasets in the appropriate folder before running the notebooks.

Required datasets:

```text
train_transaction.csv
midterm-regresi-dataset.csv
```

The datasets are not uploaded directly to GitHub because they may be large. Users should place the datasets manually in the working directory or Google Drive before running the notebooks.

### 4. Run the Notebooks

Open the notebooks using Jupyter Notebook or Google Colab, then run each section sequentially.

Recommended notebook order:

1. `fraud_detection_classification.ipynb`
2. `song_year_regression.ipynb`

---

## Requirements

The main libraries used in this project are:

```text
pandas
numpy
matplotlib
scikit-learn
tensorflow
optuna
mlflow
lime
joblib
```

---

## Conclusion

This repository successfully implements two end-to-end machine learning and deep learning projects. The first project focuses on fraud detection classification, while the second project focuses on song release year regression.

Both projects apply a complete machine learning workflow, including data preprocessing, model training, hyperparameter tuning using Optuna, evaluation using appropriate metrics, interpretation, and MLflow tracking. The experiments show that model performance can be improved through better preprocessing, appropriate evaluation metrics, and hyperparameter tuning.

---

## Author

Name: Hadi Nugroho
Class: TK-47-02
NIM: 101032330235
Course: Machine Learning and Deep Learning
