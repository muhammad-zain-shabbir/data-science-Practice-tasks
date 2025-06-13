# Data Science Internship Tasks - DevelopersHub

This repository contains my completed tasks for the Data Science & Analytics Internship at DevelopersHub Corporation.

---

# Credit Risk Prediction & Data Exploration

## 📌 Objective
This project contains five tasks:
1. **Task 1**: Explore and visualize the Iris dataset
2. **Task 2**: Predict loan approval using classification models
3. **Task 3**: Predict whether a customer will leave (churn) from a bank using classification techniques
4. **Task 4**: Estimate the insurance charges for individuals based on personal attributes using a regression model
5. **Task 5**: Predict whether a customer will accept a personal loan offer based on demographic and marketing data
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

# Task 3 Customer Churn Prediction - Bank Customers

## 📌 Objective
Predict whether a customer will leave (churn) from a bank using classification techniques.

---

## 📂 Dataset
- **Source:** Churn Modelling Dataset (Kaggle)
- Includes features such as:
  - CreditScore, Geography, Gender, Age
  - Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
  - `Exited` (target: 1 = left the bank, 0 = stayed)

---

## 🧹 Data Cleaning
- Checked for missing values (✅ none found)
- Dropped irrelevant columns: `RowNumber`, `CustomerId`, `Surname`

---

## 🔤 Feature Encoding
- Applied **Label Encoding** to `Gender` (Male=1, Female=0)
- Applied **One-Hot Encoding** to `Geography` (France dropped to avoid dummy trap)

---

## 📊 Model Preparation
- Features stored in `X`, target (`Exited`) in `y`
- Split into training and testing sets (80/20 split)

---

## 🤖 Model Training
- Model used: **Random Forest Classifier**
- Trained using `X_train`, `y_train`
- Made predictions on `X_test`

---

## 📏 Model Evaluation
- **Accuracy Score:** Displayed in notebook
- **Confusion Matrix:** Displayed to show prediction breakdown

---

## 📈 Feature Importance
- Analyzed using `model.feature_importances_`
- Visualized with `seaborn.barplot`
- Top predictors: Age, EstimatedSalary, CreditScore, Balance

---

## 🛠️ Tools Used
- **Python**, Jupyter Notebook
- Libraries: `pandas`, `numpy`, `seaborn`, `matplotlib`, `sklearn`

---

## 📁 Files in This Repo
- `customer_churn_prediction.ipynb`: Complete analysis and model training notebook
- `Churn_Modelling.csv`: Dataset used
- `README.md`: This file

---

## ✅ Skills Demonstrated
- Data preprocessing and cleaning
- Categorical encoding
- Training and evaluating classification models
- Interpreting model output and visualizing results

---

  ## Task 4 Medical Insurance Cost Prediction

## 📌 Objective

Estimate the insurance charges for individuals based on personal attributes using a regression model.

## 📂 Dataset

Source: Medical Cost Personal Dataset on Kaggle

Features include:

age, sex, bmi, children, smoker, region

charges (target variable: insurance cost)

## 🔍 Exploratory Data Analysis

Age vs Charges: Scatter plot revealed a positive trend.

BMI vs Charges: Charges tend to increase with BMI.

Smoker vs Charges: Smokers had much higher charges on average.

## 🧹 Data Preprocessing

Verified no missing values.

Applied One-Hot Encoding to categorical variables: sex, region, smoker.

## 🤖 Model Training

Model used: Linear Regression

Features selected: All numerical and encoded categorical features

Split data into 80% training and 20% testing

## 📏 Model Evaluation

Mean Absolute Error (MAE): Indicates average error in predictions

Root Mean Squared Error (RMSE): Penalizes larger errors more heavily

## 🛠️ Tools and Libraries

Python (Jupyter Notebook)

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

## 📁 Files in This Repo

insurance_cost_prediction.ipynb: Main notebook

insurance.csv: Dataset

README.md: Project overview

## ✅ Skills Demonstrated

Regression modeling

Feature encoding and visualization

# Model evaluation using MAE and RMSE

---

# Task 5 Personal Loan Acceptance Prediction

## 📌 Objective
Predict whether a customer will accept a personal loan offer based on demographic and marketing data.

---

## 📂 Dataset
- **Source:** [Bank Marketing Dataset - UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- **Format:** Semicolon-separated CSV (`bank-additional-full.csv`)
- **Target:** `y` (binary — yes/no for loan acceptance)

---

## 🔍 Exploratory Data Analysis (EDA)
- **Age vs Acceptance:** Younger users more likely to accept
- **Job Type vs Acceptance:** Admin and technician roles showed higher acceptance
- **Marital Status vs Acceptance:** Singles were more likely to accept

---

## 🧹 Data Preprocessing
- One-Hot Encoding applied to all categorical variables using `pd.get_dummies()`
- Cleaned and prepared for modeling

---

## 🤖 Model Training
- **Model Used:** Logistic Regression
- **Data Split:** 80% training, 20% testing
- **Solver:** lbfgs with increased iterations to ensure convergence

---

## 📏 Evaluation
- **Accuracy Score:** Reported for test set
- **Confusion Matrix:** Displayed to evaluate prediction categories
- **(Optional)** Classification report used for precision, recall, f1-score

---

## 💡 Key Insights
- **Age, job, and marital status** significantly influence acceptance
- Singles and certain job roles had higher acceptance rates

---

## 🛠️ Tools Used
- Python (Jupyter Notebook)
- `pandas`, `seaborn`, `matplotlib`
- `scikit-learn` for model training and evaluation

---

## 📁 Files in This Repo
- `loan_acceptance_prediction.ipynb`: Jupyter notebook with all analysis
- `bank-additional-full.csv`: Dataset (from UCI)
- `README.md`: Summary and project details


---

💡 *This project was created as part of a learning task to develop and demonstrate beginner-level data science skills.*
---
More tasks coming soon!
