import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.title("💳 Credit Card Fraud Detection using Machine Learning")

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def train_model(data):
    # Separate legitimate and fraudulent transactions
    legit = data[data["Class"] == 0]
    fraud = data[data["Class"] == 1]

    # Balance the dataset
    legit_sample = legit.sample(n=len(fraud), random_state=2)
    balanced_data = pd.concat([legit_sample, fraud], axis=0)

    # Split features and target
    X = balanced_data.drop(columns=["Class"])
    y = balanced_data["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=2
    )

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, train_accuracy, test_accuracy, X.columns

uploaded_file = st.file_uploader(
    "Upload the creditcard.csv dataset",
    type=["csv"]
)

if uploaded_file is not None:

    data = load_data(uploaded_file)

    st.success("Dataset Loaded Successfully!")

    st.write("### Dataset Preview")
    st.dataframe(data.head())

    st.write("Dataset Shape:", data.shape)

    model, train_acc, test_acc, feature_names = train_model(data)

    st.subheader("Model Accuracy")

    st.write("Training Accuracy:", round(train_acc * 100, 2), "%")
    st.write("Testing Accuracy:", round(test_acc * 100, 2), "%")

    st.subheader("Predict a Transaction")

    st.write("Enter one value for each feature.")

    input_data = []

    cols = st.columns(3)

    for i, feature in enumerate(feature_names):
        with cols[i % 3]:
            value = st.number_input(feature, value=0.0)
            input_data.append(value)

    if st.button("Predict"):

        prediction = model.predict([input_data])

        if prediction[0] == 0:
            st.success("✅ Legitimate Transaction")
        else:
            st.error("🚨 Fraudulent Transaction")