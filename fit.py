import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load dataset
data = pd.read_csv("economic_data.csv")

# Define features and target variable
X = data[["interest_rate", "inflation_rate", "gdp_growth"]]
y = data["loan_default_risk"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, predictions):.2f}")
print(f"R2 Score: {r2_score(y_test, predictions):.2f}")

# Save model
with open("economic_model.pkl", "wb") as f:
    pickle.dump(model, f)
