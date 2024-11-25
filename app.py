import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# Load the trained model
model = joblib.load('XGBoost_fraud_detection_model.pkl')

# Function to predict fraud
def predict_fraud(input_data):
    prediction = model.predict(input_data)
    return prediction[0]

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Fraud", "Fraud"], yticklabels=["Not Fraud", "Fraud"])
    ax.set_title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    st.pyplot(fig)

# Streamlit app
def app():
    st.title("Fraud Detection with Machine Learning")
    st.markdown("This app uses an XGBoost model to predict fraudulent transactions.")
    
    # Allow user to upload a dataset
    uploaded_file = st.file_uploader("Upload a CSV file for evaluation (optional):", type=["csv"])
    if uploaded_file is not None:
        # Load the uploaded dataset
        df = pd.read_csv(uploaded_file)
        st.success("Dataset uploaded successfully!")
        st.write("Preview of Uploaded Dataset:")
        st.write(df.head())
    else:
        # Load a default dataset
        df = pd.read_csv('fraud_detection_test_data.csv')
        if st.checkbox("Show Default Sample Data"):
            st.write(df.head())
    
    # Map transaction types to their encoded values
    transaction_type_mapping = {
    "CASH_IN": 0,
    "CASH_OUT": 1,
    "DEBIT": 2,
    "PAYMENT": 3,
    "TRANSFER": 4
    }

    # Sidebar input for transaction type
    st.sidebar.header("Transaction Details")
    transaction_type = st.sidebar.selectbox("Select Transaction Type", options=list(transaction_type_mapping.keys()))
    transaction_type_encoded = transaction_type_mapping[transaction_type]  # Get encoded value

    # Input fields for transaction data
    step = st.sidebar.number_input("Transaction Time (step)", min_value=0.0, value=200.0)
    amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=0.0)
    old_balance = st.sidebar.number_input("Old Balance (Origin)", min_value=0.0, value=0.0)
    new_balance = st.sidebar.number_input("New Balance (Origin)", min_value=0.0, value=0.0)
    oldbalanceDest = st.sidebar.number_input("Old Balance (Destination)", min_value=0.0, value=0.0)
    newbalanceDest = st.sidebar.number_input("New Balance (Destination)", min_value=0.0, value=0.0)

    # Input for flag status
    flag_status = st.sidebar.radio("Select Flag Status", options=["Not Flagged", "Flagged"])
    flag_value = 1 if flag_status == "Flagged" else 0  
    
    # Input data for prediction
    input_data = pd.DataFrame({
        'step': [step],
        'type': [transaction_type_encoded],
        'amount': [amount],
        'oldbalanceOrg': [old_balance],
        'newbalanceOrig': [new_balance],
        'oldbalanceDest': [oldbalanceDest],
        'newbalanceDest': [newbalanceDest],
        'isFlaggedFraud': [flag_value]
    })
    
    # Prediction button
    if st.sidebar.button("Predict"):
        prediction = predict_fraud(input_data)
        if prediction == 1:
            st.sidebar.write("🚨 **This transaction is Fraudulent!**")
        else:
            st.sidebar.write("✅ **This transaction is Not Fraudulent.**")

    # Model evaluation metrics
    if st.checkbox("Show Trained Model Metrics"):
        st.subheader("Evaluation Metrics")
        y_test = df['isFraud']  
        # Replace with your actual features
        y_pred = model.predict(df.drop(columns=['isFraud']))  
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        st.write(f"**Accuracy**: {accuracy:.4f}")
        st.write(f"**F1 Score**: {f1:.4f}")
        
        # Plot confusion matrix
        plot_confusion_matrix(y_test, y_pred)

# Run the app
if __name__ == "__main__":
    app()
