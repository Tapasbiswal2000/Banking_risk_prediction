# Banking Risk Prediction

An end-to-end machine learning project for predicting banking customer risk and ranking customers according to their probability of being classified as bad customers.

The project focuses on building a risk-ranking model for a highly imbalanced classification problem and evaluates model performance using both conventional classification metrics and business-oriented rank-ordering metrics such as **ROC-AUC, Gini coefficient, decile analysis, and cumulative bad capture**.

---

## 📌 Project Overview

Financial institutions need reliable methods to identify customers who may represent higher credit risk.

In this project, multiple machine learning classification algorithms were trained and evaluated to predict the target variable `Bad_label`.

The workflow covers:

* Data understanding and quality assessment
* Exploratory data analysis
* Feature preprocessing
* Handling numerical and categorical variables
* Train-test splitting
* Multiple baseline classification models
* Model comparison
* Rank-ordering and decile analysis
* Feature selection
* Hyperparameter tuning
* Threshold analysis
* Feature importance analysis
* Business-oriented interpretation
* Model serialization and verification

The final trained preprocessing and Gradient Boosting model are saved together as a reusable pipeline.

---

## 🎯 Business Problem

The objective is to develop a model that can assign a risk probability to each customer and rank customers from relatively higher risk to relatively lower risk.

Because the target variable is highly imbalanced, accuracy alone is not sufficient for evaluating the model.

Therefore, this project places particular emphasis on:

* ROC-AUC
* Gini coefficient
* Precision
* Recall
* F1 Score
* Decile analysis
* Cumulative bad capture

These metrics help evaluate how effectively the model separates and ranks potentially risky customers.

---

## 📊 Dataset

The dataset contains customer-level banking and credit-related information.

### Dataset characteristics

* Original features: **161**
* Training observations: **19,116**
* Test observations: **4,780**
* Target variable: `Bad_label`
* Target classes:

  * `0` → Good customer
  * `1` → Bad customer

The dataset contains both:

* Numerical features
* Categorical features

The data includes information related to areas such as:

* Credit/payment behaviour
* Account history
* Delinquency indicators
* Credit enquiries
* Account age
* Payment timing
* Customer/application characteristics

> **Note:** Raw customer-level banking data is not included in this repository.

---

## 🔄 Machine Learning Workflow

```text
Raw Banking Data
       │
       ▼
Data Understanding
       │
       ▼
Data Quality Checks
       │
       ├── Missing Values
       ├── Duplicates
       ├── Data Types
       └── Class Distribution
       │
       ▼
Exploratory Data Analysis
       │
       ▼
Train / Test Split
       │
       ▼
Feature Preprocessing
       │
       ├── Numerical Features
       └── Categorical Features
       │
       ▼
Baseline Model Evaluation
       │
       ▼
Rank-Ordering / Decile Analysis
       │
       ▼
Feature Selection
       │
       ▼
Top Model Comparison
       │
       ▼
Hyperparameter Tuning
       │
       ▼
Threshold Analysis
       │
       ▼
Final Model Evaluation
       │
       ▼
Feature Importance
       │
       ▼
Save Complete Pipeline
```

---

## 🧹 Feature Preprocessing

The project uses separate preprocessing logic for numerical and categorical variables.

The preprocessing pipeline includes operations such as:

* Missing-value handling
* Numerical preprocessing
* Categorical preprocessing
* One-hot encoding
* Feature transformation

The final feature-selection stage reduced the original feature space from:

```text
161 original features
        ↓
75 selected features
```

After preprocessing and one-hot encoding, these produced:

```text
492 transformed features
```

The complete preprocessing and model were stored together in a single pipeline to ensure consistent transformations during future predictions.

---

## 🤖 Models Evaluated

Multiple classification algorithms were evaluated during the baseline modeling stage.

### Models

* Logistic Regression
* Balanced Logistic Regression
* Decision Tree
* Random Forest
* Extra Trees
* Gradient Boosting
* XGBoost
* LightGBM
* CatBoost

The models were compared using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Gini

Rank-ordering performance was also evaluated using decile analysis.

---

## 📈 Baseline Model Comparison

| Model                        | Accuracy | Precision | Recall |     F1 | ROC-AUC |   Gini |
| ---------------------------- | -------: | --------: | -----: | -----: | ------: | -----: |
| Logistic Regression          |   0.9569 |    0.1429 | 0.0050 | 0.0096 |  0.6156 | 0.2311 |
| Balanced Logistic Regression |   0.6818 |    0.0635 | 0.4776 | 0.1121 |  0.6069 | 0.2137 |
| Decision Tree                |   0.9165 |    0.0541 | 0.0597 | 0.0567 |  0.5069 | 0.0138 |
| Random Forest                |   0.9579 |    0.0000 | 0.0000 | 0.0000 |  0.6086 | 0.2172 |
| Extra Trees                  |   0.9577 |    0.0000 | 0.0000 | 0.0000 |  0.6087 | 0.2174 |
| Gradient Boosting            |   0.9569 |    0.1429 | 0.0050 | 0.0096 |  0.6569 | 0.3138 |
| XGBoost                      |   0.9575 |    0.3333 | 0.0100 | 0.0193 |  0.5908 | 0.1815 |
| LightGBM                     |   0.9577 |    0.0000 | 0.0000 | 0.0000 |  0.6308 | 0.2616 |
| CatBoost                     |   0.9579 |    0.0000 | 0.0000 | 0.0000 |  0.6500 | 0.2999 |

Because of the strong class imbalance, the model comparison did not rely on accuracy alone.

---

## 🏆 Final Model

The final selected model is:

**Gradient Boosting Classifier**

The model was selected after comparing the evaluated models and subsequently tuning its hyperparameters.

### Final configuration

| Parameter          | Value |
| ------------------ | ----: |
| `n_estimators`     |   150 |
| `learning_rate`    |  0.05 |
| `max_depth`        |     2 |
| `min_samples_leaf` |    10 |
| `subsample`        |   0.8 |
| Selected features  |    75 |

---

## 📊 Final Model Performance

| Metric                 |                     Result |
| ---------------------- | -------------------------: |
| Accuracy               |                 **95.79%** |
| Precision              |                  **0.00%** |
| Recall                 |                  **0.00%** |
| F1 Score               |                  **0.00%** |
| ROC-AUC                |                 **0.6632** |
| Gini                   |                 **32.63%** |
| Project Benchmark Gini |                 **37.90%** |
| Benchmark Gap          | **5.27 percentage points** |

The zero precision/recall/F1 at the default classification threshold is retained as an important result rather than being hidden. This reflects the severe class imbalance and the model's conservative default classification behaviour.

---

## 📊 Rank-Ordering Performance

Since this is a highly imbalanced risk-prediction problem, rank-ordering performance is particularly important.

Customers were divided into ten deciles according to their predicted probability of being bad.

### Cumulative bad capture

| Population Considered | Bad Customers Captured |
| --------------------: | ---------------------: |
|               Top 10% |             **22.89%** |
|               Top 20% |             **35.82%** |
|               Top 30% |             **53.73%** |
|               Top 40% |             **64.18%** |
|               Top 50% |             **72.64%** |

This shows that the model provides useful risk ranking even though its default 0.50 classification threshold produces very few positive classifications.

---

## 🎚️ Threshold Analysis

Different probability thresholds were evaluated to understand the trade-off between precision and recall.

| Threshold | Accuracy | Precision | Recall |     F1 | Predicted Bad Customers |
| --------: | -------: | --------: | -----: | -----: | ----------------------: |
|      0.05 |   0.7467 |    0.0778 | 0.4627 | 0.1331 |                   1,196 |
|      0.10 |   0.9293 |    0.1040 | 0.0896 | 0.0963 |                     173 |
|      0.15 |   0.9492 |    0.1111 | 0.0299 | 0.0471 |                      54 |
|      0.20 |   0.9548 |    0.1053 | 0.0100 | 0.0182 |                      19 |
|      0.25 |   0.9571 |    0.1667 | 0.0050 | 0.0097 |                       6 |
|      0.50 |   0.9579 |    0.0000 | 0.0000 | 0.0000 |                       0 |

The threshold analysis demonstrates that the classification threshold significantly affects the number of customers identified as bad.

For risk-ranking applications, the predicted probability itself and the resulting customer ranking can therefore be more informative than relying only on the default 0.50 threshold.

---

## 🔍 Feature Importance

Feature importance was analyzed after training the final Gradient Boosting model.

The top original features included:

| Rank | Feature                        | Importance |
| ---: | ------------------------------ | ---------: |
|    1 | `feature_7`                    |     10.51% |
|    2 | `max_dpd_mean`                 |      4.62% |
|    3 | `feature_12`                   |      4.52% |
|    4 | `feature_52`                   |      4.22% |
|    5 | `dpd_0_29_count_mean`          |      4.08% |
|    6 | `maximum_days_to_closure`      |      3.70% |
|    7 | `average_days_to_last_payment` |      3.61% |
|    8 | `average_account_age_days`     |      3.41% |
|    9 | `enquiry_amount_last_180_days` |      3.07% |
|   10 | `days_entry_to_opened`         |      3.06% |

The top 10 original features contributed approximately **44.81%** of cumulative feature importance.

The top 20 contributed approximately **70.26%**, while the top 30 contributed approximately **84.05%**.

---

## 💡 Business Insights

Comparison of the important numerical features between good and bad customer groups showed several observable differences.

For example:

* `max_dpd_mean` was higher among bad customers.
* `enquiry_amount_last_180_days` was higher among bad customers.
* `enquiries_last_365_days` was higher among bad customers.
* `days_entry_to_opened` was slightly higher among bad customers.
* Average account age was lower among bad customers.
* Average days to last payment was lower among bad customers.

These observations represent **associations within the dataset** and should not be interpreted as causal relationships.

Some anonymized variables such as `feature_7`, `feature_12`, and `feature_52` were highly important, but their business definitions were not available. Therefore, no unsupported business interpretation has been assigned to them.

---

## 💾 Model Saving

The final preprocessing and Gradient Boosting model were saved as a single `joblib` pipeline:

```text
model/banking_risk_gb_pipeline.pkl
```

This allows the complete preprocessing and prediction workflow to be reused without manually reconstructing the preprocessing steps.

### Model verification

The saved pipeline was loaded again and tested against the original model.

Result:

```text
Predictions identical: True
Maximum probability difference: 0.0
```

Therefore, the serialized pipeline successfully reproduces the original model predictions.

---

## 📁 Project Structure

```text
Banking_Risk_Prediction/
│
├── data/
│   └── ...
│
├── model/
│   └── banking_risk_gb_pipeline.pkl
│
├── notebook/
│   └── banking_risk_prediction.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```

Raw customer-level data is intentionally excluded from version control.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <your-repository-url>
```

Move into the project directory:

```bash
cd Banking_Risk_Prediction
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Open the notebook:

```bash
jupyter notebook
```

Then open:

```text
notebook/banking_risk_prediction.ipynb
```

The notebook contains the complete workflow from data exploration through final model evaluation.

---

## 📦 Model Prediction Example

Once the saved pipeline is loaded:

```python
import joblib

model = joblib.load(
    "model/banking_risk_gb_pipeline.pkl"
)

probabilities = model.predict_proba(new_customer_data)[:, 1]

predictions = (probabilities >= 0.50).astype(int)
```

The probability represents the model's estimated probability of the positive class (`Bad_label = 1`).

For a risk-ranking workflow, customers can be sorted by this probability:

```python
risk_scores = model.predict_proba(new_customer_data)[:, 1]

new_customer_data["risk_probability"] = risk_scores

ranked_customers = new_customer_data.sort_values(
    "risk_probability",
    ascending=False
)
```

---

## ⚠️ Limitations

* The final Gini of **32.63%** is below the project benchmark of **37.9%**.
* The dataset is highly imbalanced.
* Default-threshold classification produces zero recall for the positive class.
* Some features are anonymized and cannot be given meaningful business interpretations.
* Feature importance represents model importance, not causality.
* Threshold selection should ultimately depend on the business cost of false positives and false negatives.
* Model performance may change when applied to a different population or future data.

---

## 🚀 Future Improvements

Potential future improvements include:

* More extensive hyperparameter optimization
* Cross-validation focused on Gini/ROC-AUC
* Probability calibration
* Cost-sensitive learning
* More systematic threshold optimization
* SHAP-based model explainability
* Population Stability Index (PSI) monitoring
* Model drift monitoring
* Feature stability analysis
* Time-based validation
* API deployment using FastAPI
* Docker containerization
* CI/CD automation
* Cloud deployment
* Production monitoring

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* XGBoost
* LightGBM
* CatBoost
* Jupyter Notebook
* Joblib

---

## 📌 Project Status

**Modeling and evaluation: Completed ✅**

The final model has been trained, evaluated, serialized, and verified.

Future development can focus on productionization through:

**FastAPI → Docker → CI/CD → Deployment → Monitoring**

---

## 👨‍💻 Author

**Tapas Biswal**

Aspiring Data Scientist | Python | SQL | Machine Learning | Data Analytics | AI/ML

GitHub: `https://github.com/Tapasbiswal2000`
