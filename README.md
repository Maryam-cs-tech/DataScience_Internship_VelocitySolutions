# DataScience_Internship_VelocitySolutions
# 🏥 Medical Cost Prediction – Dual Regression Model

This project aims to predict both **medical expenses** and **insurance premium** based on a customer's demographic and health-related information using machine learning regression techniques.

---

## 📌 Project Overview

This project was developed as part of a hands-on data science internship. The goal was to build a predictive system that can estimate both `expenses` and `premium` values for health insurance customers using features like age, gender, BMI, region, etc.

---

## 📁 Dataset

- **Source**: [`imtkaggleteam/health-insurance-dataset`](https://www.kaggle.com/datasets/imtkaggleteam/health-insurance-dataset)
- **Features include**:
  - `age`, `gender`, `bmi`, `children`, `discount_eligibility`, `region`
  - `expenses` and `premium` (targets)
  - `bmi_category` (derived feature for better prediction)

---

## 🔄 Workflow Summary

### ✅ Week 1: EDA & Setup
- Loaded and explored the dataset
- Performed data cleaning, missing value handling, and initial visualizations
- Created visual plots (box plots, count plots, correlation heatmaps) to understand relationships

### ✅ Week 2: Feature Engineering & Preprocessing
- Created `bmi_category` from raw BMI values
- One-hot encoded categorical columns like `region` and `bmi_category`
- Scaled numerical values where necessary
- Split data into features (`X`) and targets (`y_exp`, `y_prem`)
- Applied `train_test_split` for model validation

### ✅ Week 3: Model Training & Prediction
- Trained two separate **Linear Regression** models:
  - One for predicting `expenses_capped`
  - One for predicting `premium_capped`
- Evaluated model performance using:
  - **RMSE** (Root Mean Squared Error)
  - **MAE** (Mean Absolute Error)
  - **R²** (Coefficient of Determination)
- Created a final prediction function that:
  - Accepts new customer input
  - Reconstructs the feature vector
  - Returns predicted **medical expenses** and **insurance premium**

---

## 🔍 Model Performance

| Metric     | Expenses Model | Premium Model |
|------------|----------------|----------------|
| RMSE       | 0.00           |     0.00       |
| MAE        | 0.00           |     0.00       |
| R² Score   | 1.00           |     1.00       |


## 🧠 Technologies Used

- **Python 3**
- **pandas**, **numpy**, **matplotlib**, **seaborn**
- **scikit-learn** for modeling
- **Jupyter Notebook** for experimentation

---

## 🚀 Final Prediction Function

The final function `predict_costs()` accepts a dictionary of customer attributes and returns predicted values:

```python
new_customer = {
    'age': 40,
    'gender': 1,
    'bmi': 29.5,
    'children': 2,
    'discount_eligibility': 1,
    'region_northwest': 0,
    'region_southeast': 0,
    'region_southwest': 1,
    'bmi_cat_underweight': 0,
    'bmi_cat_normal': 0,
    'bmi_cat_overweight': 1,
    'bmi_cat_obese': 0
}
