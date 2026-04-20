# 🩺 Diabetes Prediction Engine: Production ML Pipeline

## Overview

This repository contains an end-to-end machine learning pipeline that predicts the risk of diabetes based on medical metrics. Instead of a simple Jupyter Notebook script, this project focuses on **production-grade feature engineering** and robust deployment practices using Scikit-Learn's `Pipeline` and `ColumnTransformer`, served via a Streamlit web interface.

## Key Machine Learning Practices Demonstrated

* **Zero Data Leakage:** All preprocessing steps (imputation, scaling) are bundled within a Scikit-Learn `Pipeline` and tuned via `GridSearchCV`.
* **Advanced Imputation:**
  * **Median Imputation:** Automatically catches physiological zeros (e.g., Blood Pressure = 0) and imputes the training median.
  * **Missing Indicators:** Uses `SimpleImputer(add_indicator=True)` to handle missing `Insulin` data by creating a boolean flag and filling missing values with `-999`, allowing the Decision Tree to learn from the *absence* of the test.
* **Stateful Preprocessing:** The backend UI remains completely decoupled from the data math. The deployed `.pkl` artifact handles all data routing and transformation natively.

## 📂 Repository Structure

```text
```text
diabetes-prediction-project/
│
├── data/
│   └── diabetes.csv                            # Raw dataset
├── notebooks/
│   └── diabetes_full_production_grade.ipynb    # EDA and GridSearchCV pipeline
├── models/
│   └── diabetes_production_model.pkl           # Serialized Pipeline & Model
├── app.py                                      # Streamlit frontend GUI
├── requirements.txt                            # Exact environment dependencies
├── .gitignore                                  # Ignores venv and non-project files
└── README.md                                   # Project documentation
```

## 🚀 Installation and Setup

1. **Clone the repository:**

   ```bash
   git clone <https://github.com/Sohailmateen/diabetes.git>
   cd diabetes-prediction-project
   ```
2. **Create a virtual environment (Recommended):**

   ```bash
   python -m venv venv
   # On Windows use: venv\Scripts\activate
   # On Mac/Linux use: source venv/bin/activate
   ```
3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Streamlit App:**

   ```bash
   streamlit run app.py
   ```

## 🧪 Testing the Model

Once the app is running locally at `http://localhost:8501`, you can use the following sample profiles to test the model's pipeline logic:

**Sample 1: Healthy Baseline (Low Risk)**

* Pregnancies: 1 | Glucose: 85 | Blood Pressure: 66
* Insulin: 60 | BMI: 22.5 | DPF: 0.350 | Age: 24

**Sample 2: Missing Data Handling (High Risk)**
*Test the missing indicator logic by leaving Insulin at 0.*

* Pregnancies: 5 | Glucose: 168 | Blood Pressure: 82
* Insulin: 0 | BMI: 34.2 | DPF: 0.750 | Age: 45

**Sample 3: Messy Data Robustness**
*Test the pipeline's ability to impute missing medians dynamically.*

* Pregnancies: 2 | Glucose: 0 | Blood Pressure: 0
* Insulin: 0 | BMI: 28.0 | DPF: 0.250 | Age: 33

## 🛠️ Technologies Used

* **Machine Learning:** Python, Scikit-learn, Pandas, NumPy
* **Model Serialization:** Joblib
* **Frontend UI:** Streamlit
