import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from pathlib import Path

# ---------------------------------------
# Load Header Image (GitHub-safe)
# ---------------------------------------
header_path = Path("header.png")

if header_path.exists():
    st.image(str(header_path), use_container_width=True)
else:
    st.error("❌ header.png not found. Make sure it is uploaded to your GitHub repo.")

# ---------------------------------------
# Subtitle
# ---------------------------------------
st.markdown("""
<div style='text-align: center; margin-top: -15px;'>
    <p style='font-size: 20px; color: #666;'>A data-driven tool to explore trends and predict LinkedIn use ✨</p>
</div>
""", unsafe_allow_html=True)

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
# Load & Train (Hidden)
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
# Label Dictionaries
# ---------------------------------------

marital_labels = {0: "Not Married", 1: "Married"}
gender_labels = {0: "Male", 1: "Female"}

income_labels_full = {
    1: "Less than $10,000",
    2: "10 to under $20,000",
    3: "20 to under $30,000",
    4: "30 to under $40,000",
    5: "40 to under $50,000",
    6: "50 to under $75,000",
    7: "75 to under $100,000",
    8: "100 to under $150,000",
    9: "$150,000 or more"
}

education_labels_full = {
    1: "Less than high school",
    2: "High school incomplete",
    3: "High school graduate",
    4: "Some college, no degree",
    5: "Two-year associate degree",
    6: "Bachelor’s degree",
    7: "Some postgraduate schooling",
    8: "Postgraduate / Professional degree"
}

# ---------------------------------------
# Tabs Layout
# ---------------------------------------
tab1, tab2 = st.tabs(["🔮 Predict LinkedIn Use", "📊 Explore Trends"])


# ================================================================
# 🔮 TAB 1 — PREDICTION
# ================================================================
with tab1:
    st.header("Tell us about this person")

    st.sidebar.header("User Inputs")

    # Sidebar Input Controls
    income = st.sidebar.selectbox(
        "Household Income",
        options=list(income_labels_full.keys()),
        format_func=lambda x: income_labels_full[x]
    )

    education_choice = st.sidebar.selectbox(
        "Education Level",
        options=list(education_labels_full.keys()),
        format_func=lambda x: education_labels_full[x]
    )

    age = st.sidebar.slider("Age", 1, 97, 40)

    parent_choice = st.sidebar.selectbox(
        "Parent?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x else "No"
    )

    married_choice = st.sidebar.selectbox(
        "Married?",
        options=[0, 1],
        format_func=lambda x: marital_labels[x]
    )

    female_choice = st.sidebar.selectbox(
        "Gender",
        options=[0, 1],
        format_func=lambda x: gender_labels[x]
    )

    st.subheader("User Profile Summary")
    st.info(f"""
    **Income:** {income_labels_full[income]}  
    **Education:** {education_labels_full[education_choice]}  
    **Age:** {age}  
    **Parent:** {"Yes" if parent_choice else "No"}  
    **Married:** {marital_labels[married_choice]}  
    **Gender:** {gender_labels[female_choice]}  
    """)

    # Prediction
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
    classification = "LinkedIn User" if prob >= 0.5 else "Non-User"


    st.success(f"🌟 **Estimated LinkedIn Use Probability: {percentage}%**")
    st.write("This estimate is based on modeling patterns observed in survey data.")
    st.subheader(f"Prediction: {classification}")


# ================================================================
# 📊 TAB 2 — EXPLORATORY ANALYSIS
# ================================================================
with tab2:
    st.header("Explore Broader Social Patterns 💫")

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

    # ---------------------------
    # Distribution of LinkedIn Use
    # ---------------------------
    if plot_choice == "Distribution of LinkedIn Use":
        sns.countplot(data=ss, x="sm_li", ax=ax)
        ax.set_xticklabels(["Non-User (0)", "LinkedIn User (1)"])
        ax.set_xlabel("LinkedIn Use Category")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of LinkedIn Use")

        for c in ax.containers:
            ax.bar_label(c, padding=3)

    # ---------------------------
    # LinkedIn Use by Parent
    # ---------------------------
    elif plot_choice == "LinkedIn Use by Parent":
        sns.barplot(data=ss, x="parent", y="sm_li", ax=ax)
        ax.set_xticklabels(["Not a Parent (0)", "Parent (1)"])
        ax.set_xlabel("Parental Status")
        ax.set_ylabel("Average LinkedIn Usage")
        ax.set_title("LinkedIn Use by Parental Status")

    # ---------------------------
    # LinkedIn Use by Married
    # ---------------------------
    elif plot_choice == "LinkedIn Use by Married":
        sns.barplot(data=ss, x="married", y="sm_li", ax=ax)
        ax.set_xticklabels(["Not Married (0)", "Married (1)"])
        ax.set_xlabel("Marital Status")
        ax.set_ylabel("Average LinkedIn Usage")
        ax.set_title("LinkedIn Use by Marital Status")

    # ---------------------------
    # LinkedIn Use by Gender
    # ---------------------------
    elif plot_choice == "LinkedIn Use by Gender":
        sns.barplot(data=ss, x="female", y="sm_li", ax=ax)
        ax.set_xticklabels(["Male (0)", "Female (1)"])
        ax.set_xlabel("Gender")
        ax.set_ylabel("Average LinkedIn Usage")
        ax.set_title("LinkedIn Use by Gender")

    # ---------------------------
    # LinkedIn Use by Income
    # ---------------------------
    elif plot_choice == "LinkedIn Use by Income":
        sns.barplot(data=ss, x="income", y="sm_li", ax=ax)
        ax.set_xticklabels(list(income_labels_full.values()), rotation=45, ha="right")
        ax.set_xlabel("Income")
        ax.set_ylabel("Average LinkedIn Usage")
        ax.set_title("LinkedIn Use by Income")

    # ---------------------------
    # LinkedIn Use by Education
    # ---------------------------
    elif plot_choice == "LinkedIn Use by Education":
        sns.barplot(data=ss, x="education", y="sm_li", ax=ax)
        ax.set_xticklabels(list(education_labels_full.values()), rotation=45, ha="right")
        ax.set_xlabel("Education")
        ax.set_ylabel("Average LinkedIn Usage")
        ax.set_title("LinkedIn Use by Education")

    # ---------------------------
    # LinkedIn Use by Age
    # ---------------------------
    elif plot_choice == "LinkedIn Use by Age":
        sns.lineplot(data=ss, x="age", y="sm_li", ax=ax)
        ax.set_xlabel("Age")
        ax.set_ylabel("Average LinkedIn Usage")
        ax.set_title("LinkedIn Use by Age")

    st.pyplot(fig)
