import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Income Classification ML App",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Income Classification using Machine Learning")
st.write(
    "This application compares five classification models "
    "for predicting whether income is <=50K or >50K."
)

st.divider()

# --------------------------------------------------
# Load trained models
# --------------------------------------------------

@st.cache_resource
def load_models():

    models = {
        "Logistic Regression": joblib.load(
            "model/logistic_regression.pkl"
        ),

        "Decision Tree": joblib.load(
            "model/decision_tree.pkl"
        ),

        "K-Nearest Neighbors": joblib.load(
            "model/knn.pkl"
        ),

        "Naive Bayes": joblib.load(
            "model/naive_bayes.pkl"
        ),

        "Random Forest": joblib.load(
            "model/random_forest.pkl"
        )
    }

    scaler = joblib.load("model/scaler.pkl")

    return models, scaler


models, scaler = load_models()

# --------------------------------------------------
# Upload test data
# --------------------------------------------------

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)

if uploaded_file is None:

    st.info(
        "Please upload the test_data.csv file generated "
        "during the ML experiment."
    )

    st.stop()

# Read uploaded CSV
df = pd.read_csv(uploaded_file)

st.success(
    f"Dataset loaded successfully: {df.shape[0]} rows × "
    f"{df.shape[1]} columns"
)

# --------------------------------------------------
# Dataset preview
# --------------------------------------------------

st.subheader("Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)

# --------------------------------------------------
# Prepare data
# --------------------------------------------------

if "income" not in df.columns:

    st.error(
        "The uploaded dataset must contain an 'income' column."
    )

    st.stop()

# Separate features and target
X = df.drop("income", axis=1)
y = df["income"]

# --------------------------------------------------
# Encode categorical variables
# --------------------------------------------------

categorical_columns = X.select_dtypes(
    include="object"
).columns.tolist()

X_encoded = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True
)

X_encoded = X_encoded.astype(float)

# --------------------------------------------------
# Match the training feature structure
# --------------------------------------------------

# The trained models expect 97 features.
# Reindexing ensures the uploaded test data has
# exactly the same feature structure.


# Align test features with the features used during model training
X_encoded = X_encoded.reindex(
    columns=scaler.feature_names_in_,
    fill_value=0
)


# Scale data using the saved scaler
X_scaled = scaler.transform(X_encoded)

# --------------------------------------------------
# Convert target values
# --------------------------------------------------

y = y.astype(int)

# --------------------------------------------------
# Model selection
# --------------------------------------------------

st.header("2. Select Classification Model")

selected_model = st.selectbox(
    "Choose a model",
    list(models.keys())
)

model = models[selected_model]

# --------------------------------------------------
# Prediction
# --------------------------------------------------

y_pred = model.predict(X_scaled)

# Probability for AUC
if hasattr(model, "predict_proba"):

    y_probability = model.predict_proba(X_scaled)[:, 1]

else:

    y_probability = None

# --------------------------------------------------
# Evaluation metrics
# --------------------------------------------------

accuracy = accuracy_score(y, y_pred)

precision = precision_score(
    y,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y,
    y_pred,
    zero_division=0
)

mcc = matthews_corrcoef(
    y,
    y_pred
)

if y_probability is not None:

    auc = roc_auc_score(
        y,
        y_probability
    )

else:

    auc = 0.0

# --------------------------------------------------
# Display metrics
# --------------------------------------------------

st.header("3. Model Evaluation")

col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

col1.metric(
    "Accuracy",
    f"{accuracy:.4f}"
)

col2.metric(
    "AUC",
    f"{auc:.4f}"
)

col3.metric(
    "Precision",
    f"{precision:.4f}"
)

col4.metric(
    "Recall",
    f"{recall:.4f}"
)

col5.metric(
    "F1 Score",
    f"{f1:.4f}"
)

col6.metric(
    "MCC",
    f"{mcc:.4f}"
)

# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

st.header("4. Confusion Matrix")

cm = confusion_matrix(
    y,
    y_pred
)

fig, ax = plt.subplots()

ax.imshow(cm)

ax.set_title(
    f"Confusion Matrix - {selected_model}"
)

ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels([
    "<=50K",
    ">50K"
])

ax.set_yticklabels([
    "<=50K",
    ">50K"
])

for i in range(2):

    for j in range(2):

        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

st.pyplot(fig)

# --------------------------------------------------
# Classification Report
# --------------------------------------------------

st.header("5. Classification Report")

report = classification_report(
    y,
    y_pred,
    target_names=[
        "<=50K",
        ">50K"
    ],
    zero_division=0
)

st.text(report)

# --------------------------------------------------
# Prediction Summary
# --------------------------------------------------

st.header("6. Prediction Summary")

prediction_counts = pd.Series(
    y_pred
).value_counts().sort_index()

summary_df = pd.DataFrame({
    "Income Class": [
        "<=50K",
        ">50K"
    ],
    "Predicted Count": [
        prediction_counts.get(0, 0),
        prediction_counts.get(1, 0)
    ]
})

st.dataframe(
    summary_df,
    use_container_width=True
)

st.success(
    f"{selected_model} prediction and evaluation completed successfully."
)