import streamlit as st
import pandas as pd
import joblib

model = joblib.load(
    "models/attrition_model.pkl"
)

st.title(
    "Employee Attrition Prediction"
)

age = st.slider(
    "Age",
    18,
    60,
    30
)

monthly_income = st.number_input(
    "Monthly Income",
    1000,
    50000,
    5000
)

distance = st.slider(
    "Distance From Home",
    1,
    30,
    5
)

years_company = st.slider(
    "Years At Company",
    0,
    40,
    5
)

overtime = st.selectbox(
    "OverTime",
    ["Yes","No"]
)

job_role = st.selectbox(
    "Job Role",
    [
        "Sales Executive",
        "Research Scientist",
        "Laboratory Technician"
    ]
)

sample = pd.DataFrame({

'Age':[age],
'BusinessTravel':['Travel_Rarely'],
'DailyRate':[800],
'Department':['Research & Development'],
'DistanceFromHome':[distance],
'Education':[3],
'EducationField':['Life Sciences'],
'EnvironmentSatisfaction':[3],
'Gender':['Male'],
'HourlyRate':[60],
'JobInvolvement':[3],
'JobLevel':[2],
'JobRole':[job_role],
'JobSatisfaction':[3],
'MaritalStatus':['Single'],
'MonthlyIncome':[monthly_income],
'MonthlyRate':[15000],
'NumCompaniesWorked':[2],
'OverTime':[overtime],
'PercentSalaryHike':[15],
'PerformanceRating':[3],
'RelationshipSatisfaction':[3],
'StockOptionLevel':[1],
'TotalWorkingYears':[10],
'TrainingTimesLastYear':[3],
'WorkLifeBalance':[3],
'YearsAtCompany':[years_company],
'YearsInCurrentRole':[4],
'YearsSinceLastPromotion':[1],
'YearsWithCurrManager':[4]

})


if st.button("Predict Attrition"):

    pred = model.predict(sample)
    prob = model.predict_proba(sample)[:, 1]

    if pred[0] == 1:
        st.error(
            f"Employee is likely to leave.\n\nRisk Score: {prob[0]:.2%}"
        )
    else:
        st.success(
            f"Employee is likely to stay.\n\nRisk Score: {prob[0]:.2%}"
        )
