# Example JSON request format expected by the CML model deployment:
# 
# The model expects input data to be structured as follows:
#
# {
#   "parameters": {
#     "interest_rate": 2.5,   # Interest rate in percentage (e.g., 2.5%).
#     "inflation_rate": 3.2,  # Inflation rate in percentage (e.g., 3.2%).
#     "gdp_growth": 1.8       # GDP growth rate in percentage (e.g., 1.8%).
#   }
# }
#
# When making an API request to the deployed model, ensure the input follows this format.
# The model will return a prediction for "loan_default_risk" based on these economic indicators.



import pickle
import pandas as pd
import cml.models_v1 as models  # Required for CML deployment

# Load trained model
with open("economic_model.pkl", "rb") as f:
    model = pickle.load(f)

@models.cml_model
def predict(args):
    # Extract parameters from request
    params = args.get("parameters", {})
    interest_rate = float(params.get("interest_rate", 0))
    inflation_rate = float(params.get("inflation_rate", 0))
    gdp_growth = float(params.get("gdp_growth", 0))

    # Prepare input for prediction
    input_data = pd.DataFrame({
        "interest_rate": [interest_rate],
        "inflation_rate": [inflation_rate],
        "gdp_growth": [gdp_growth]
    })

    # Make prediction
    result = model.predict(input_data)

    # Return formatted response
    return {"loan_default_risk": float(result[0])}
