# Framingham Heart Disease Prediction

An end-to-end machine learning project that predicts the risk of developing coronary heart disease within 10 years using the Framingham Heart Study dataset.

## About the Project

I built this project to practice applying machine learning to a real-world healthcare dataset. Instead of only training a model and checking its accuracy, I worked through the complete classification workflow — from cleaning the data and preparing the features to cross-validation, hyperparameter tuning, threshold analysis, and final evaluation on an unseen test set.

One of the main challenges in this dataset is class imbalance. Most observations belong to the "No Heart Disease" class, while the number of positive cases is much smaller. Because of this, accuracy alone does not give a complete picture of the model's performance. I therefore used precision, recall, F1-score, and ROC-AUC along with accuracy.

## Dataset

The original dataset contains **4,240 records** and **16 columns**. After removing observations containing missing values, **3,751 records** remained.

The target variable is `TenYearCHD`:

- `0` → No coronary heart disease within 10 years
- `1` → Coronary heart disease within 10 years

### Class Distribution

| Class | Percentage |
|---|---:|
| No Heart Disease | 84.75% |
| Heart Disease | 15.25% |

The model uses 14 input features, including age, gender, smoking status, cigarettes per day, blood pressure, cholesterol, BMI, heart rate, glucose, diabetes, hypertension, and previous stroke.

## Project Workflow

```text
Framingham Dataset
       ↓
Data Cleaning
       ↓
Feature & Target Separation
       ↓
Train / Validation / Test Split
       ↓
Feature Selection
       ↓
Feature Standardization
       ↓
Logistic Regression
       ↓
Stratified 5-Fold Cross-Validation
       ↓
Hyperparameter Tuning
       ↓
Threshold Analysis
       ↓
Final Evaluation on Unseen Test Data
```

## Machine Learning Approach

I used **Logistic Regression** as the classification algorithm.

The Scikit-learn pipeline includes:

- `SelectKBest` for feature selection
- `StandardScaler` for standardization
- `LogisticRegression` for classification

### Cross-Validation

I used **Stratified 5-Fold Cross-Validation** so that the class distribution was preserved across folds.

Mean cross-validation ROC-AUC before tuning: **0.7367**

### Hyperparameter Tuning

`GridSearchCV` was used to evaluate different combinations of the regularization parameter `C`, solver, class weighting, and number of selected features.

The best configuration was:

```text
C = 0.1
Solver = lbfgs
Class weight = None
Features = All 14
```

Best cross-validation ROC-AUC: **0.7373**

Class weighting was tested because of the imbalance in the target variable, but the best-performing configuration according to cross-validation selected `class_weight=None`.

## Train, Validation and Test Data

The cleaned dataset was divided into development and final test data. The development data was further split into model-training and threshold-validation sets:

```text
3,751 records
       │
       ├── 3,000 → Development data
       │      │
       │      ├── 2,400 → Model training
       │      │
       │      └── 600 → Threshold validation
       │
       └── 751 → Final test set
```

The validation set was used to select the classification threshold. The final test set was kept separate and used only for final evaluation.

## Threshold Analysis

A classification model produces a probability for the positive class. Normally, a threshold of `0.50` is used to convert probabilities into class predictions.

In this project, I tested multiple thresholds and compared their precision, recall, and F1-score. The threshold producing the highest F1-score on the validation set was:

**0.20**

This threshold was then used for the final evaluation on the unseen test set.

## Final Results

| Metric | Score |
|---|---:|
| Accuracy | **73.77%** |
| Precision | **28.19%** |
| Recall | **46.09%** |
| F1-Score | **34.98%** |
| ROC-AUC | **70.07%** |

### Confusion Matrix

```text
                 Predicted
                No CHD    CHD

Actual No CHD     501     135
Actual CHD         62      53
```

- True Negatives = 501
- False Positives = 135
- False Negatives = 62
- True Positives = 53

The model correctly identified **53 of the 115 actual positive cases** in the test set.

## What I Learned

This project helped me understand that model accuracy is not always enough, especially when working with an imbalanced dataset.

Some of the main things I learned were:

- Cleaning and preparing a real-world dataset
- Separating training, validation, and test data
- Building a Scikit-learn pipeline
- Standardizing features
- Performing feature selection
- Using Stratified K-Fold Cross-Validation
- Performing hyperparameter tuning with GridSearchCV
- Understanding class imbalance
- Working with prediction probabilities
- Selecting a classification threshold
- Evaluating models using precision, recall, F1-score, and ROC-AUC
- Interpreting a confusion matrix

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook / Google Colab

## Project Structure

```text
framingham-heart-disease-prediction/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── Framingham_Heart_Disease_Prediction.ipynb
│
├── data/
│   └── framingham.csv
│
└── images/
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── precision_recall_curve.png
    └── threshold_analysis.png
```

## Disclaimer

This project is intended for educational and machine learning demonstration purposes only. It is not a clinical diagnostic system and should not be used for medical decision-making.
