import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ---------------------------------------
# App Header
# ---------------------------------------
st.title("LinkedIn Usage Prediction Tool")
st.write("Use this tool to estimate how likely someone is to use LinkedIn based on demographic characteristics.")

# ---------------------------------------
# Helper Functions
# ---------------------------------------
def clean_sm(x):
    return np.where(x == 1, 1, 0)

@st.cache_data
def load_data():
    return pd.read_csv("social_media_usage.csv")

def build_feature_df(s):
    ss = pd.DataFrame({
        "sm_li": clean_sm(s["web1h"]),
        "income": s["income"].where(s["income"] <= 9),
        "education": s["educ2"].where(s["educ2"] <= 8),
        "parent": clean_sm(s["par"]),
        "married": clean_sm(s["marital"]),
        "female": clean_sm(s["gender"]),
        "age": s["age"].where(s["age"] <= 98),
    }).dropna()
    return ss

# ---------------------------------------
# Load & Train (Hidden From Users)
# ---------------------------------------
s = load_data()
ss = build_feature_df(s)

X = ss[["income", "education", "age", "parent", "married", "female"]]
y = ss["sm_li"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=15217
)

log_reg = LogisticRegression(class_weight="balanced")
log_reg.fit(x_train, y_train)

# ---------------------------------------
# Sidebar User Inputs
# ---------------------------------------
st.header("Estimate Someone's Likelihood of Using LinkedIn")

st.sidebar.header("User Inputs")

# Income dropdown
income_labels = {
    1: "Less than $10,000",
    2: "$10,000–$19,999",
    3: "$20,000–$29,999",
    4: "$30,000–$39,999",
    5: "$40,000–$49,999",
    6: "$50,000–$74,999",
    7: "$75,000–$99,999",
    8: "$100,000–$149,999",
    9: "$150,000 or more"
}

income = st.sidebar.selectbox(
    "Household Income",
    options=list(income_labels.keys()),
    format_func=lambda x: income_labels[x]
)

# Education dropdown
education_labels = {
    1: "Less than high school",
    2: "High school incomplete",
    3: "High school graduate",
    4: "Some college, no degree",
    5: "Two-year associate degree",
    6: "Bachelor’s degree",
    7: "Some postgraduate schooling",
    8: "Postgraduate or professional degree"
}

education_choice = st.sidebar.selectbox(
    "Education Level",
    options=list(education_labels.keys()),
    format_func=lambda x: education_labels[x]
)

# Age slider
age = st.sidebar.slider("Age", 1, 97, 40)

# Parent
parent_choice = st.sidebar.selectbox(
    "Parent?",
    options=[0, 1],
    format_func=lambda x: "Yes" if x else "No"
)

# Married
married_choice = st.sidebar.selectbox(
    "Married?",
    options=[0, 1],
    format_func=lambda x: "Married" if x else "Not Married"
)

# Gender
female_choice = st.sidebar.selectbox(
    "Gender",
    options=[0, 1],
    format_func=lambda x: "Female" if x else "Male"
)

# ---------------------------------------
# Prediction
# ---------------------------------------
person = pd.DataFrame({
    "income": [income],
    "education": [education_choice],
    "age": [age],
    "parent": [parent_choice],
    "married": [married_choice],
    "female": [female_choice]
})

prob = log_reg.predict_proba(person)[0][1]
percentage = round(prob * 100, 1)

st.subheader("Predicted Likelihood of LinkedIn Use")
st.metric("Probability", f"{percentage}%")

st.write("This estimate is based on modeling patterns observed in national survey data.")

# ---------------------------------------
# EXPLORATORY ANALYSIS (now at the bottom)
# ---------------------------------------
st.header("Explore Broader Social Trends")

plot_choice = st.selectbox(
    "Choose a visualization",
    [
        "Distribution of LinkedIn Use",
        "LinkedIn Use by Parent",
        "LinkedIn Use by Married",
        "LinkedIn Use by Gender",
        "LinkedIn Use by Income",
        "LinkedIn Use by Education",
        "LinkedIn Use by Age",
    ]
)

fig, ax = plt.subplots()

if plot_choice == "Distribution of LinkedIn Use":
    sns.countplot(data=ss, x="sm_li", ax=ax)
    ax.set_title("Distribution of LinkedIn Use")

elif plot_choice == "LinkedIn Use by Parent":
    sns.barplot(data=ss, x="parent", y="sm_li", ax=ax)
    ax.set_title("LinkedIn Use by Parental Status")

elif plot_choice == "LinkedIn Use by Married":
    sns.barplot(data=ss, x="married", y="sm_li", ax=ax)
    ax.set_title("LinkedIn Use by Marital Status")

elif plot_choice == "LinkedIn Use by Gender":
    sns.barplot(data=ss, x="female", y="sm_li", ax=ax)
    ax.set_title("LinkedIn Use by Gender")

elif plot_choice == "LinkedIn Use by Income":
    sns.barplot(data=ss, x="income", y="sm_li", ax=ax)
    ax.set_title("LinkedIn Use by Income")

elif plot_choice == "LinkedIn Use by Education":
    sns.barplot(data=ss, x="education", y="sm_li", ax=ax)
    ax.set_title("LinkedIn Use by Education")

elif plot_choice == "LinkedIn Use by Age":
    sns.lineplot(data=ss, x="age", y="sm_li", ax=ax)
    ax.set_title("LinkedIn Use by Age")

st.pyplot(fig)
