import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Bank Loan Prediction Dashboard", layout="wide")

# ---------------------------
# LOAD OR GENERATE DATA
# ---------------------------
def load_data():
    try:
        df = pd.read_csv("UniversalBank.csv")
    except FileNotFoundError:
        # Create synthetic data if CSV not present
        np.random.seed(42)
        df = pd.DataFrame({
            "ID": range(1, 501),
            "Personal Loan": np.random.randint(0, 2, 500),
            "Age": np.random.randint(20, 65, 500),
            "Experience": np.random.randint(0, 40, 500),
            "Income": np.random.randint(20, 200, 500),
            "ZIP Code": np.random.randint(10000, 99999, 500),
            "Family": np.random.randint(1, 5, 500),
            "CCAvg": np.random.uniform(0.5, 10, 500),
            "Education": np.random.randint(1, 4, 500),
            "Mortgage": np.random.randint(0, 300, 500),
            "Securities": np.random.randint(0, 2, 500),
            "CDAccount": np.random.randint(0, 2, 500),
            "Online": np.random.randint(0, 2, 500),
            "CreditCard": np.random.randint(0, 2, 500),
        })
    return df

df = load_data()

# ---------------------------
# DATA EXPLORATION
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📊 Insights", "🤖 Model Training", "📁 Upload & Predict"])

if page == "📊 Insights":
    st.title("Customer Insights Dashboard")

    # Income bins (robust handling)
    try:
        df["IncomeDecile"] = pd.qcut(df["Income"], 10, labels=False, duplicates="drop")
    except Exception:
        df["IncomeDecile"] = pd.cut(df["Income"], 10, labels=False)

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.histogram(df, x="Age", color="Personal Loan", barmode="group", title="Age vs Loan Status")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.box(df, x="Education", y="Income", color="Personal Loan", title="Income by Education")
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(df, x="CCAvg", y="Income", color="Personal Loan",
                      size="Family", title="Spending vs Income")
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.histogram(df, x="Family", color="Personal Loan", barmode="group", title="Family Size vs Loan")
    st.plotly_chart(fig4, use_container_width=True)

    corr = df.drop(columns=["ID", "ZIP Code"]).corr()
    fig5 = px.imshow(corr, text_auto=True, title="Feature Correlation Heatmap")
    st.plotly_chart(fig5, use_container_width=True)

# ---------------------------
# MODEL TRAINING
# ---------------------------
elif page == "🤖 Model Training":
    st.title("Model Comparison and Evaluation")

    X = df.drop(columns=["ID", "Personal Loan", "ZIP Code"])
    y = df["Personal Loan"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosted Tree": GradientBoostingClassifier(random_state=42)
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results.append({
            "Algorithm": name,
            "Training Accuracy": accuracy_score(y_train, y_pred_train),
            "Testing Accuracy": accuracy_score(y_test, y_pred_test),
            "Precision": precision_score(y_test, y_pred_test),
            "Recall": recall_score(y_test, y_pred_test),
            "F1 Score": f1_score(y_test, y_pred_test),
            "AUC": roc_auc_score(y_test, y_prob)
        })

    results_df = pd.DataFrame(results)
    st.dataframe(results_df.style.highlight_max(color="lightgreen", axis=0))

    # ROC curves
    st.subheader("ROC Curves")
    plt.figure(figsize=(7, 5))
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    st.pyplot(plt)

    # Confusion matrices
    for name, model in models.items():
        st.subheader(f"Confusion Matrix: {name}")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.heatmap(confusion_matrix(y_train, model.predict(X_train)),
                    annot=True, fmt="d", cmap="Blues", ax=axes[0])
        axes[0].set_title("Training")
        sns.heatmap(confusion_matrix(y_test, model.predict(X_test)),
                    annot=True, fmt="d", cmap="Greens", ax=axes[1])
        axes[1].set_title("Testing")
        st.pyplot(fig)

    # Feature importance
    for name, model in models.items():
        if hasattr(model, "feature_importances_"):
            feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            st.subheader(f"Feature Importance: {name}")
            st.bar_chart(feat_imp)

# ---------------------------
# PREDICT NEW DATA
# ---------------------------
else:
    st.title("Upload New Data for Prediction")
    file = st.file_uploader("Upload a CSV file", type=["csv"])

    if file:
        new_df = pd.read_csv(file)
        model = RandomForestClassifier(random_state=42)
        X = df.drop(columns=["ID", "Personal Loan", "ZIP Code"])
        y = df["Personal Loan"]
        model.fit(X, y)

        new_X = new_df[X.columns.intersection(df.columns)]
        new_df["Predicted_PersonalLoan"] = model.predict(new_X)
        st.dataframe(new_df.head())

        csv = new_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predicted Data", data=csv, file_name="predicted_results.csv")
