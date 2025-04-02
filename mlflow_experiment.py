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

# Start the MLflow run
with mlflow.start_run() as run:
    try:
        # Log some parameters for clarity
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("model_type", "LinearRegression")
        
        # Define and train your model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions and evaluate
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Log evaluation metrics
        mlflow.log_metric("mean_squared_error", mse)
        mlflow.log_metric("r2_score", r2)
        
        # Log the model along with its signature
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name="EconomicRiskModel"
        )
        
        print("MLflow experiment completed and model registered successfully!")
        
    except Exception as e:
        # Capture the traceback and log it as an artifact
        error_trace = traceback.format_exc()
        print("An error occurred during the MLflow run:")
        print(error_trace)
        mlflow.set_tag("run_status", "failed")
        
        # Write the traceback to a temporary file and log it as an artifact
        error_file = "error_trace.txt"
        with open(error_file, "w") as f:
            f.write(error_trace)
        mlflow.log_artifact(error_file)
        
        # Re-raise the exception so that the notebook cell shows the error
        raise