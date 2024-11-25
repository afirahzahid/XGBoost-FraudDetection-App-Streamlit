# Fraud Detection in Online Transactions

## Project Overview
This project addresses the critical challenge of detecting fraudulent financial transactions using advanced machine learning techniques. The system leverages an **XGBoost model** trained on a highly imbalanced dataset of over 6 million records. Additionally, a user-friendly **Streamlit app** allows real-time evaluation of transactions, making it suitable for practical deployment in financial institutions.

## Live Demo
[Fraud Detection App](https://frauddetectionxgboost.streamlit.app/)

## Features
- **Fraud Prediction**: Predict fraudulent transactions using a trained XGBoost model.
- **Interactive Dashboard**: Upload custom data or evaluate sample data through a Streamlit interface.
- **Performance Metrics**: Display model accuracy, F1 score, and a visual confusion matrix.
- **Transaction Input Panel**: Provide transaction details (e.g., type, amount, balances) for real-time predictions.

## File Details
- **`.gitignore`**: Ensures temporary files, logs, and sensitive data aren't tracked.
- **`XGBoost_fraud_detection_model.pkl`**: Trained model for fraud detection.
- **`app.py`**: Streamlit application allowing users to:
  - Upload datasets for batch fraud prediction.
  - Input transaction details manually for real-time predictions.
  - Visualize evaluation metrics (accuracy, F1 score) and confusion matrix.
- **`dataetset_EDA_and_ML_Modeling.ipynb`**: Jupyter Notebook detailing data exploration, preprocessing, and different model training approaches such as Logistic Regression, KNN, Random Forest, Naive Bayes, and XGBoost.
- **`fraud_detection_test_data.csv`**: Sample dataset with transaction records for testing the app.
- **`requirements.txt`**: List of libraries (e.g., Streamlit, XGBoost) required to run the project.

## Example Usage
### Predicting Real-Time Transactions
Input transaction details via the app’s sidebar (e.g., transaction type, amount, balances) and click "Predict" to determine if the transaction is fraudulent.

### Batch Prediction
Upload a CSV file containing transaction records. The app will process the data and provide the prediction for the batch dataset.

