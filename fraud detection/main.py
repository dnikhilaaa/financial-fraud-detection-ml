import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load and preprocess data
df = pd.read_csv("transactions.csv")
df = pd.get_dummies(df, columns=["transaction_type"], drop_first=True)
X = df.drop(columns=["transaction_id", "is_fraud"])
y = df["is_fraud"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Isolation Forest": IsolationForest(contamination=0.1)
}

# Evaluate models
for name, model in models.items():
    if name == "Isolation Forest":
        model.fit(X_train)
        preds = model.predict(X_test)
        preds = [1 if i == -1 else 0 for i in preds]
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    print(f"\nModel: {name}")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))
