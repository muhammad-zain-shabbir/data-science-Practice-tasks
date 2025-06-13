# Data Science Internship Tasks - DevelopersHub

This repository contains my completed tasks for the Data Science & Analytics Internship at DevelopersHub Corporation.

---

# Credit Risk Prediction & Data Exploration

## 📌 Objective
This project contains two tasks:
1. **Task 1**: Explore and visualize the Iris dataset
2. **Task 2**: Predict loan approval using classification models

---

## 🧪 Task 1: Exploring and Visualizing a Simple Dataset (Iris)

### 🎯 Goal
Understand how to read, summarize, and visualize a dataset using pandas, matplotlib, and seaborn.

### 📂 Dataset
- **Source:** Built-in Iris dataset from the `seaborn` library

### 🔍 What Was Done
- Loaded the dataset using `seaborn.load_dataset()`
- Displayed dataset structure using:
  - `.shape`
  - `.columns`
  - `.head()`
- Created visualizations:
  - **Scatter Plot** (petal length vs width by species)
  - **Histogram** (distribution of sepal length)
  - **Box Plot** (sepal length by species)

### 🛠️ Tools Used
- Python (Jupyter Notebook)
- pandas, seaborn, matplotlib

### 📁 Files
- `iris_visualization.ipynb`: Notebook for Task 1 (Iris exploration)


---

# Task 2 Credit Risk Prediction

## 📌 Objective
The goal of this project is to predict whether a loan applicant is likely to default on a loan. This is done by training a classification model using a real-world loan dataset.

## 📂 Dataset
- **Source:** [Kaggle - Loan Prediction Problem Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- The dataset includes features like:
  - `LoanAmount`
  - `Education`
  - `ApplicantIncome`
  - Other demographic and financial details

## 🧹 Tasks Performed

### 1. Data Cleaning
- Identified and handled missing values using:
  - **Mean** for numerical columns (e.g., `LoanAmount`)
  - **Mode** for categorical columns (e.g., `Gender`, `Married`, etc.)

### 2. Exploratory Data Analysis (EDA)
- Visualized key features to explore relationships:
  - **Loan Amount Distribution** (Histogram)
  - **Education vs Loan Status** (Bar Chart)
  - **Applicant Income by Loan Status** (Box Plot)

### 3. Model Building
- Selected key features: `ApplicantIncome`, `LoanAmount`
- Applied **classification models**:
  - **Logistic Regression**
  - **Decision Tree Classifier**

### 4. Model Evaluation
- Evaluated performance using:
  - **Accuracy Score**
  - **Confusion Matrix**

## 📊 Results
- **Logistic Regression Accuracy:** ~83%
- **Confusion Matrix:** Demonstrated good separation of approved vs. not approved

## 🛠️ Tools Used
- **Python** (Jupyter Notebook)
- **pandas** (for data handling)
- **seaborn & matplotlib** (for data visualization)
- **scikit-learn** (for model training and evaluation)

## 📁 Files in This Repo
- `credit_risk_prediction.ipynb`: Main Jupyter notebook with all code and analysis
- `loan_data.csv`: Dataset used for training and testing
- `README.md`: Project summary and instructions

## ✅ Skills Practiced
- Data cleaning and handling missing values
- Visualizing and understanding dataset features
- Binary classification using ML models
- Evaluating model performance

---

💡 *This project was created as part of a learning task to develop and demonstrate beginner-level data science skills.*
---
More tasks coming soon!
