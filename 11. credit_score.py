import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load data
data = pd.read_csv("C:/Users/saran/OneDrive/Documents/ML/DATASET/CREDITSCORE.csv")

# Select features and target variable
X = data[["Annual_Income", "Monthly_Inhand_Salary",
          "Num_Bank_Accounts", "Num_Credit_Card",
          "Interest_Rate", "Num_of_Loan",
          "Delay_from_due_date", "Num_of_Delayed_Payment",
          "Credit_Mix", "Outstanding_Debt",
          "Credit_History_Age", "Monthly_Balance"]]
y = data['Credit_Score']

# Convert 'Credit_Mix' to numerical using Label Encoding
label_encoder = LabelEncoder()
X['Credit_Mix'] = label_encoder.fit_transform(X['Credit_Mix'])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Make predictions for new data
new_data = pd.DataFrame({
    "Annual_Income": [50000],
    "Monthly_Inhand_Salary": [4000],
    "Num_Bank_Accounts": [2],
    "Num_Credit_Card": [1],
    "Interest_Rate": [0.05],
    "Num_of_Loan": [1],
    "Delay_from_due_date": [10],
    "Num_of_Delayed_Payment": [0],
    "Credit_Mix": ["Good"],  # Assuming it's a categorical variable
    "Outstanding_Debt": [10000],
    "Credit_History_Age": [5],
    "Monthly_Balance": [300]
})

# Encode 'Credit_Mix' in the new data
new_data['Credit_Mix'] = label_encoder.transform(new_data['Credit_Mix'])

# Make prediction
predicted_score = model.predict(new_data)
print("Predicted Credit Score:", predicted_score)