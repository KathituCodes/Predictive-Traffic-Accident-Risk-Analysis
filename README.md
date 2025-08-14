# Predictive Traffic Accident Risk Analysis and Deployment with Streamlit

## Overview
This project builds a machine learning model to predict high-risk traffic accidents in Kenya using historical data. It classifies accidents as "high risk" or "low risk" based on factors like victims, deaths, and time. The model uses Logistic Regression with SMOTE for class imbalance. Deployed via Streamlit for real-time predictions.

## Problem Statement
Traffic accidents cause high fatalities in Kenya due to poor infrastructure and reactive measures. This tool enables proactive risk assessment for better resource allocation.

## Objectives
- Develop ML model to classify accident risk.
- Deploy interactive Streamlit app for user inputs and predictions.

## Dataset
- Source: TRAFFIC ACCIDENTS DATA.xlsx (78 records).
- Features: Date, Accident Spot, Area, County, Road/Highway, Cause, Victims, Deaths, Time.
- Preprocessing: Cleaned missing data, encoded categories, engineered features (e.g., High Risk column via victim thresholding).
- Challenges: Class imbalance (77 low-risk vs. 1 high-risk); addressed with SMOTE.

## Exploratory Data Analysis (EDA)
- Insights: Higher frequencies on certain days/locations; multi-victim accidents more high-risk.
- Visualizations: (Add plots from notebook if available).
- Confirmed dataset suitability for modeling.

## Model
- Algorithm: Logistic Regression (class_weight='balanced') + SMOTE.
- Evaluation: Accuracy, Precision, Recall, F1-Score.
- Alternative: Decision Tree for risk levels (Low, Moderate, High) based on victims, deaths, time.
- Exported: road_risk_model.pkl.

## Installation
1. Clone repo: `git clone https://github.com/yourusername/repo.git`
2. Install dependencies: `pip install -r requirements.txt` (pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, joblib, streamlit).
3. Run notebook: `jupyter notebook project.ipynb`

## Usage
- Run model training/evaluation in `project.ipynb`.
- Test predictions:
  ```python
  import joblib
  clf = joblib.load('road_risk_model.pkl')
  # Input: [victim_category, deaths, time_category]
  prediction = clf.predict([[5, 4, 3]])  # e.g., passengers, 4 deaths, night -> High
  ```
- Deploy: `streamlit run app.py` (assumes app.py for interface).

## Deployment
Streamlit app:
- Inputs: Victim category (0-9), Deaths, Time category (0-3).
- Outputs: Risk classification (Low/Moderate/High).
- Error handling for invalid inputs.

## Validation
- Instructor feedback: Relevant, feasible; focus on interpretability and imbalance.

## Impact
Scalable tool for traffic safety; reduces fatalities via data-driven decisions.
