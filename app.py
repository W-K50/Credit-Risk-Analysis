import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer

# Load model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.load_model(r'C:\Users\hp\Desktop\Credit Risk Analysis\xgb_model.json')

# Load and prepare training data
train_df = pd.read_csv(r'C:\Users\hp\Desktop\Credit Risk Analysis\cs-training.csv', index_col=0)
X_train = train_df.drop('SeriousDlqin2yrs', axis=1)

# Feature engineering for training set
def feature_engineering(df):
    df['DebtRatioPerIncome'] = df['DebtRatio'] / (df['MonthlyIncome'] + 1)
    df['AgeBucket'] = pd.cut(df['age'], bins=[0, 30, 50, 100], labels=[0, 1, 2])
    return df

X_train = feature_engineering(X_train)

# Imputer fit
imputer = SimpleImputer(strategy='median')
imputer.fit(X_train)

# Streamlit App
st.title("Credit Risk Prediction")
st.markdown("Enter customer details to predict the likelihood of defaulting on a loan.")

# Input form
with st.form("user_form"):
    RevolvingUtilizationOfUnsecuredLines = st.slider("Revolving Utilization", 0.0, 2.0, 0.5)
    age = st.slider("Age", 18, 100, 35)
    NumberOfTime30_59DaysPastDueNotWorse = st.slider("30-59 Days Past Due", 0, 20, 0)
    DebtRatio = st.slider("Debt Ratio", 0.0, 10.0, 1.5)
    MonthlyIncome = st.number_input("Monthly Income", min_value=0, value=5000)
    NumberOfOpenCreditLinesAndLoans = st.slider("Open Credit Lines & Loans", 0, 50, 5)
    NumberOfTimes90DaysLate = st.slider("90 Days Late", 0, 20, 0)
    NumberRealEstateLoansOrLines = st.slider("Real Estate Loans or Lines", 0, 10, 1)
    NumberOfTime60_89DaysPastDueNotWorse = st.slider("60-89 Days Past Due", 0, 20, 0)
    NumberOfDependents = st.slider("Number of Dependents", 0, 20, 0)
    submitted = st.form_submit_button("Predict")

# After submit
if submitted:
    user_input = pd.DataFrame({
        'RevolvingUtilizationOfUnsecuredLines': [RevolvingUtilizationOfUnsecuredLines],
        'age': [age],
        'NumberOfTime30-59DaysPastDueNotWorse': [NumberOfTime30_59DaysPastDueNotWorse],
        'DebtRatio': [DebtRatio],
        'MonthlyIncome': [MonthlyIncome],
        'NumberOfOpenCreditLinesAndLoans': [NumberOfOpenCreditLinesAndLoans],
        'NumberOfTimes90DaysLate': [NumberOfTimes90DaysLate],
        'NumberRealEstateLoansOrLines': [NumberRealEstateLoansOrLines],
        'NumberOfTime60-89DaysPastDueNotWorse': [NumberOfTime60_89DaysPastDueNotWorse],
        'NumberOfDependents': [NumberOfDependents]
    })

    # Feature engineering
    user_input = feature_engineering(user_input)

    # Ensure same column order
    user_input = user_input[X_train.columns]

    # Impute
    user_input_imputed = pd.DataFrame(imputer.transform(user_input), columns=X_train.columns)

    # Predict
    pred_proba = model.predict_proba(user_input_imputed)[:, 1]
    pred_label = "🔴 High Risk" if pred_proba[0] > 0.5 else "🟢 Low Risk"

    st.subheader("Prediction Results")
    st.write(f"**Probability of Default**: `{pred_proba[0]:.4f}`")
    st.write(f"**Risk Category**: **{pred_label}**")
