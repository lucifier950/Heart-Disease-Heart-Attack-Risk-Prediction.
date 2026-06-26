# ❤️ Heart Disease & Heart Attack Risk Prediction

> An end-to-end machine learning project that predicts cardiovascular risk from clinical and lifestyle data — covering the full data-science lifecycle from raw data to evaluated, cross-validated models.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green)
![pandas](https://img.shields.io/badge/pandas-Data-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

---

## 📌 Overview

Heart disease is one of the leading causes of mortality worldwide. Early, data-driven identification of at-risk individuals enables timely intervention and personalized care. This project builds and evaluates supervised machine-learning models that predict heart disease / heart-attack risk from patient demographics, clinical measurements, and lifestyle factors.

The repository contains **two complementary studies**, demonstrating the workflow on both a clean clinical benchmark and a larger, noisier real-world dataset:

| Notebook | Dataset | Size | Focus |
|----------|---------|------|-------|
| [`fdaprojectcode.ipynb`](fdaprojectcode.ipynb) | UCI Heart Disease | 303 patients | Rigorous EDA + multi-model comparison (LogReg, Random Forest, XGBoost) with cross-validation |
| [`Completeprojectcode.ipynb`](Completeprojectcode.ipynb) | Heart Attack Risk ([`heartriskanalysis.csv`](heartriskanalysis.csv)) | 8,763 patients | Feature engineering + class-imbalance handling (SMOTE) + threshold tuning |

---

## 🎯 Objectives

- Perform thorough **exploratory data analysis** (univariate, bivariate, and multivariate) to understand the drivers of cardiac risk.
- Apply robust **data preprocessing**: missing-value imputation, outlier handling (IQR capping), encoding, and feature scaling.
- **Engineer domain-informed features** (Mean Arterial Pressure, a composite Lifestyle Score, an Age–Cholesterol Index).
- Train, tune, and **compare multiple classifiers**, and validate results with stratified K-fold cross-validation.
- Address **class imbalance** using SMOTE and decision-threshold optimization.

---

## 🧠 Study 1 — UCI Heart Disease (Multi-Model Comparison)

**Dataset:** UCI Heart Disease dataset, loaded programmatically via the [`ucimlrepo`](https://pypi.org/project/ucimlrepo/) package. Features include age, sex, chest-pain type (`cp`), resting blood pressure, cholesterol, max heart rate (`thalach`), ST depression (`oldpeak`), and more. The multi-class target was binarized into **disease / no-disease**.

**Pipeline**
1. Missing-value imputation (mode for categorical `ca` / `thal`).
2. One-hot encoding of categorical features + target binarization.
3. Outlier detection & capping via the Interquartile Range (IQR) method.
4. Comprehensive EDA — distributions, correlation heatmap, box/violin plots, pairplots, and 3-variable interaction plots.
5. Feature scaling with `StandardScaler` and an 80/20 stratified train/test split.
6. Training and evaluation of three models + Stratified K-Fold cross-validation.

**Results (hold-out test set)**

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|:--------:|:---------:|:------:|:--------:|
| **Logistic Regression** | **0.885** | **0.862** | **0.893** | **0.877** |
| Random Forest | 0.869 | 0.833 | 0.893 | 0.862 |
| XGBoost | 0.852 | 0.828 | 0.857 | 0.842 |

> **Logistic Regression** was the top performer, achieving **88.5% accuracy** and a **0.877 F1-score** — a strong, well-balanced result on this benchmark.

---

## 🧪 Study 2 — Heart Attack Risk (Feature Engineering & Imbalance)

**Dataset:** A larger, real-world heart-attack-risk dataset of **8,763 patients** with clinical, demographic, and lifestyle attributes (cholesterol, blood pressure, BMI, triglycerides, smoking, exercise, stress, sleep, diet, income, etc.).

**Highlights**
- **Data cleaning:** split `Blood Pressure` into systolic/diastolic, dropped identifiers, mapped `Diet` to ordinal values, one-hot encoded categorical fields.
- **Feature engineering:** created three new clinically-motivated features —
  - **Mean Arterial Pressure (MAP)**
  - a composite **Lifestyle Score**
  - an **Age–Cholesterol Index**
- **Class imbalance** addressed with **SMOTE** oversampling on the training set.
- **Threshold tuning** to improve minority-class (high-risk) recall.
- **Random Forest** classifier with `class_weight='balanced'`, validated via 5-fold cross-validation (mean F1 ≈ 0.58).

This study illustrates the realistic challenge of modeling noisy, weakly-correlated lifestyle data — and the engineering techniques (resampling, threshold adjustment, feature creation) used to squeeze out signal.

---

## 📊 Visualizations

Exploratory and model-evaluation plots are saved in the [`plots/`](plots/) directory — including correlation heatmaps, feature distributions, confusion matrices, and feature-importance charts.

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Data:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn`, `xgboost`
- **Imbalance handling:** `imbalanced-learn` (SMOTE)
- **Data source:** `ucimlrepo`
- **Environment:** Jupyter Notebook / Google Colab

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
https://github.com/lucifier950/Heart-Disease-Heart-Attack-Risk-Prediction..git
cd Heart-Disease-Heart-Attack-Risk-Prediction.
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn ucimlrepo jupyter
```

### 3. Run the notebooks
```bash
jupyter notebook
```
Open either `fdaprojectcode.ipynb` or `Completeprojectcode.ipynb` and run the cells top to bottom.

> **Note:** The notebooks were developed in Google Colab. If running locally, update any `/content/...` file paths (e.g. for `heartriskanalysis.csv`) to point to the local file.

---

## 📁 Repository Structure

```
fdaproject/
├── fdaprojectcode.ipynb        # Study 1 — UCI dataset, multi-model comparison
├── Completeprojectcode.ipynb   # Study 2 — feature engineering + SMOTE
├── heartriskanalysis.csv       # Heart attack risk dataset (8,763 records)
├── plots/                      # Saved EDA & evaluation figures
└── README.md
```

---

## 🔑 Key Takeaways

- Built a **complete, reproducible ML pipeline** — from raw data through EDA, preprocessing, feature engineering, modeling, and validation.
- Demonstrated **model selection** by benchmarking Logistic Regression, Random Forest, and XGBoost with cross-validated metrics.
- Applied practical techniques for **class imbalance** (SMOTE) and **decision-threshold optimization**.
- Achieved **88.5% accuracy / 0.877 F1** on the UCI benchmark with a well-tuned, interpretable Logistic Regression model.

---

## 📈 Future Work

- Hyperparameter tuning via `GridSearchCV` / `Optuna`.
- Model explainability with **SHAP** values.
- Deploy the best model as an interactive web app (Streamlit / Flask).

---

## 👤 Author

**Advik Rajvansh**

[GitHub](https://github.com/lucifier950)

---

⭐ *If you found this project useful or interesting, consider giving it a star!*
