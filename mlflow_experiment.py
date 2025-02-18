import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from mlflow.models import infer_signature

# Load dataset
data = pd.read_csv("economic_data.csv")

# Define features and target variable
X = data[["interest_rate", "inflation_rate", "gdp_growth"]]
y = data["loan_default_risk"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure MLflow is set to track a new experiment
mlflow.set_experiment("demo_experiment")

# Start an MLflow run
with mlflow.start_run():
    # Define model
    model = LinearRegression()

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate model
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Log metrics
    mlflow.log_metric("mean_squared_error", mse)
    mlflow.log_metric("r2_score", r2)

    # Log model with a registered model name
    registered_model_name = "EconomicRiskModel"
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        registered_model_name=registered_model_name
    )

print("MLflow experiment completed and model registered successfully!")
