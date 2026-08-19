
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("rounded_hours_student_scores.csv")

    return df


df = load_data()


# ============================================================
# TRAIN MACHINE LEARNING MODEL
# ============================================================

X = df[["Hours"]]
y = df["Scores"]

model = LinearRegression()

model.fit(X, y)

y_pred = model.predict(X)


# ============================================================
# MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(y, y_pred)

mse = mean_squared_error(y, y_pred)

r2 = r2_score(y, y_pred)


# ============================================================
# TITLE
# ============================================================

st.title("🎓 Student Score Predictor")

st.write(
    "A Machine Learning application that predicts "
    "student scores based on study hours."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📌 Project Information")

st.sidebar.write(
    """
    **Machine Learning Algorithm**

    Linear Regression

    **Input**

    Study Hours

    **Output**

    Predicted Student Score
    """
)

st.sidebar.divider()

st.sidebar.subheader("Dataset Information")

st.sidebar.write(
    f"Number of Records: {len(df)}"
)

st.sidebar.write(
    f"Minimum Study Hours: {df['Hours'].min()}"
)

st.sidebar.write(
    f"Maximum Study Hours: {df['Hours'].max()}"
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🔮 Prediction",
        "📊 Data Analysis",
        "🤖 Model Performance"
    ]
)


# ============================================================
# TAB 1 — PREDICTION
# ============================================================

with tab1:

    st.header("🔮 Predict Student Score")

    st.write(
        "Enter the number of hours a student studies "
        "to predict their expected score."
    )

    hours = st.slider(
        "Select Study Hours",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.1
    )

    st.write(
        f"### Selected Study Hours: **{hours} hours**"
    )

    if st.button(
        "🔮 Predict Score",
        use_container_width=True
    ):

        prediction = model.predict(
            np.array([[hours]])
        )[0]

        # Keep prediction between 0 and 100
        prediction = max(0, min(100, prediction))

        st.success(
            f"### Predicted Score: {prediction:.2f} / 100"
        )

        # Performance message

        if prediction >= 75:

            st.info(
                "🌟 Excellent expected performance!"
            )

        elif prediction >= 50:

            st.info(
                "👍 Good expected performance."
            )

        else:

            st.warning(
                "📚 The student may need additional "
                "study time."
            )


# ============================================================
# TAB 2 — DATA ANALYSIS
# ============================================================

with tab2:

    st.header("📊 Dataset Analysis")

    st.subheader("Dataset Preview")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader("📈 Study Hours vs Student Scores")

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.scatter(
        df["Hours"],
        df["Scores"],
        label="Actual Scores"
    )

    ax.plot(
        df["Hours"],
        y_pred,
        linewidth=2,
        label="Regression Line"
    )

    ax.set_xlabel(
        "Study Hours"
    )

    ax.set_ylabel(
        "Student Score"
    )

    ax.set_title(
        "Study Hours vs Student Scores"
    )

    ax.legend()

    st.pyplot(fig)

    st.subheader("📋 Statistical Summary")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.subheader("🔗 Correlation")

    correlation = df["Hours"].corr(
        df["Scores"]
    )

    st.write(
        f"Correlation between Study Hours "
        f"and Scores: **{correlation:.4f}**"
    )


# ============================================================
# TAB 3 — MODEL PERFORMANCE
# ============================================================

with tab3:

    st.header("🤖 Model Performance")

    st.write(
        "The Linear Regression model is evaluated "
        "using MAE, MSE and R² Score."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "MAE",
            f"{mae:.2f}"
        )

    with col2:

        st.metric(
            "MSE",
            f"{mse:.2f}"
        )

    with col3:

        st.metric(
            "R² Score",
            f"{r2:.2f}"
        )

    st.divider()

    st.subheader("📐 Linear Regression Equation")

    st.write(
        f"**Score = {model.coef_[0]:.2f} × Hours "
        f"+ {model.intercept_:.2f}**"
    )

    st.subheader("📊 Actual vs Predicted Scores")

    comparison = pd.DataFrame(
        {
            "Study Hours": df["Hours"],
            "Actual Score": df["Scores"],
            "Predicted Score": y_pred
        }
    )

    st.dataframe(
        comparison,
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Student Score Prediction | "
    "Machine Learning Project | "
    "Linear Regression"
)
