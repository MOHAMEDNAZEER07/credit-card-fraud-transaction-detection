import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Load data
data = pd.read_csv('creditcard.csv')

# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Select features and target
X = data.drop(columns="Class", axis=1)
y = data["Class"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# Streamlit app
st.title("💳 Credit Card Fraud Detection System")
st.markdown(
    """
    Welcome to the **Credit Card Fraud Detection System**. This app uses a machine learning model to predict fraudulent transactions based on historical data.
    """)

# Display model accuracy
st.subheader("Model Performance")
with st.container():
    cols = st.columns(2)
    cols[0].metric("Training Accuracy", f"{train_acc:.2%}")
    cols[1].metric("Testing Accuracy", f"{test_acc:.2%}")

# File uploader
st.subheader("🔍 Upload Your Data")
uploaded_file = st.file_uploader("Upload a CSV file with transaction features", type="csv")

if uploaded_file is not None:
    with st.spinner("Processing your file..."):
        # Read the uploaded file
        input_data = pd.read_csv(uploaded_file)
        
        # Ensure the input data does not have the 'Class' column and matches the feature columns
        input_data = input_data.drop(columns=['Class'], errors='ignore')
        expected_columns = X.columns.tolist()
        if all(col in input_data.columns for col in expected_columns):
            input_data = input_data[expected_columns]  # Reorder columns to match training data
            predictions = model.predict(input_data)
            input_data['Prediction'] = predictions
            input_data['Result'] = input_data['Prediction'].apply(lambda x: 'Legitimate' if x == 0 else 'Fraudulent')

            # Fraudulent Transactions Analysis
            fraudulent_transactions = input_data[input_data['Result'] == 'Fraudulent']
            total_transactions = len(input_data)
            fraudulent_transactions_count = len(fraudulent_transactions)
            fraud_percentage = (fraudulent_transactions_count / total_transactions) * 100

            # Select relevant columns for display
            columns_to_display = ['Time', 'Amount', 'Result']
            fraudulent_transactions_display = fraudulent_transactions[columns_to_display]

            # Display Fraud Analysis
            with st.expander("📊 Fraud Analysis"):
                st.metric("Total Transactions", total_transactions)
                st.metric("Fraudulent Transactions", fraudulent_transactions_count)
                st.metric("Percentage of Fraudulent Transactions", f"{fraud_percentage:.2f}%")
                st.subheader("Fraudulent Transactions Data")
                st.dataframe(fraudulent_transactions_display)

                # Optionally, allow downloading the filtered table
                csv_download = fraudulent_transactions_display.to_csv(index=False)
                st.download_button(
                    label="Download Fraudulent Transactions",
                    data=csv_download,
                    file_name="fraudulent_transactions.csv",
                    mime="text/csv",
                )

            # Visualizations
            with st.expander("📈 Visualizations"):
                # Pie Chart
                labels = ['Legitimate', 'Fraudulent']
                sizes = [len(legit), len(fraud)]
                fig1, ax1 = plt.subplots()
                ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999'])
                ax1.axis('equal')
                st.pyplot(fig1)

                # Boxplot
                fig2, ax2 = plt.subplots()
                sns.boxplot(x='Class', y='Amount', data=data, ax=ax2, palette='Set2')
                ax2.set_title("Transaction Amount Distribution")
                ax2.set_xticklabels(['Legitimate', 'Fraudulent'])
                st.pyplot(fig2)

                # Bar Chart: Fraudulent vs Legitimate Transactions
                fig3, ax3 = plt.subplots()
                data['Class'].value_counts().plot(kind='bar', ax=ax3, color=['#66b3ff', '#ff9999'])
                ax3.set_title("Fraudulent vs Legitimate Transactions")
                ax3.set_xticklabels(['Legitimate', 'Fraudulent'], rotation=0)
                ax3.set_ylabel("Count")
                st.pyplot(fig3)

                # Correlation Heatmap
                fig4, ax4 = plt.subplots(figsize=(10, 8))
                corr_matrix = data.corr()
                sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=ax4)
                ax4.set_title("Feature Correlation Heatmap")
                st.pyplot(fig4)

                # Histogram: Transaction Amounts
                fig5, ax5 = plt.subplots()
                sns.histplot(data=data, x='Amount', hue='Class', kde=True, ax=ax5, palette={0: 'green', 1: 'red'})
                ax5.set_title("Transaction Amounts Distribution")
                ax5.set_xlabel("Amount")
                st.pyplot(fig5)

                # Line Chart: Fraudulent Transactions Over Time
                if 'Time' in input_data.columns:
                    fraud_over_time = fraudulent_transactions.groupby('Time').size()
                    fig6, ax6 = plt.subplots()
                    fraud_over_time.plot(kind='line', ax=ax6, color='red')
                    ax6.set_title('Fraudulent Transactions Over Time')
                    ax6.set_xlabel('Time')
                    ax6.set_ylabel('Fraudulent Transactions')
                    st.pyplot(fig6)
        else:
            st.error("Uploaded file does not match the expected format. Please ensure the columns match the training data.")
else:
    st.info("Please upload a CSV file to see predictions.")
