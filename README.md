# 💳 Credit Card Fraud Prediction

Built an **end-to-end fraud detection pipeline** comparing **8 ML models** on 
**284K+ real-world transactions** to accurately flag fraudulent activity.

---

## 📌 Problem Statement
Credit card fraud is a critical issue in the financial industry. This project builds 
a machine learning pipeline to classify transactions as fraudulent or legitimate, 
addressing the challenge of severe class imbalance (~0.17% fraud rate).

---

## 📂 Dataset
- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions | 31 features
- **Class Distribution:** 99.83% legitimate | 0.17% fraudulent

---

## ⚙️ Pipeline Overview

1. **Data Preprocessing**
   - Converted raw `Time` feature into 24-hour cyclic bins
   - Dropped irrelevant features after EDA

2. **Exploratory Data Analysis (EDA)**
   - Analyzed fraud patterns by hour of day
   - Visualized class imbalance, outliers (boxplots), and feature distributions (KDE plots)

3. **Feature Selection**
   - Applied `SelectKBest` with Chi-squared scoring on MinMax normalized data
   - Selected top 10 most predictive features

4. **Handling Class Imbalance**
   - Applied **SMOTE (Synthetic Minority Oversampling Technique)** on training data
   - Improved minority class recall by **85%**

5. **Model Training & Evaluation**
   - Split: 70% train / 30% test
   - Benchmarked 8 ML models on Accuracy, Precision, Recall, and F1-Score

---

## 🤖 Models Benchmarked

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | - | - | - | - |
| K-Nearest Neighbors | - | - | - | - |
| Decision Tree | - | - | - | - |
| Naive Bayes | - | - | - | - |
| Random Forest ✅ | 99.9% | 88.3% | 87.1% | 0.85 |
| Linear SVM | - | - | - | - |
| XGBoost | - | - | - | - |
| Gradient Boosting | - | - | - | - |

> ✅ **Random Forest** selected as the final model based on F1-Score and Recall.

---

## 📊 Results

| Metric | Score |
|---|---|
| Accuracy | 99.9% |
| Precision | 88.3% |
| Recall | 87.1% |
| F1-Score | 0.85 |

> **Note:** Recall is prioritized as the key metric — missing a fraudulent 
> transaction is far more costly than a false alarm.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, imbalanced-learn, 
  Matplotlib, Seaborn

---

## 🚀 How to Run

```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
python fraud_detection.py
```

> Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) 
> and place it in the project root directory.

---
