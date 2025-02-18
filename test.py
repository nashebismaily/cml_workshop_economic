import pickle
import pandas as pd
import cml.models_v1 as models  # Required for CML deployment

# Load trained model
with open("economic_model.pkl", "rb") as f:
    model = pickle.load(f)

@models.cml_model
def predict(args):
    try:
        # Extract parameters safely
        interest_rate = float(args.get("interest_rate", 0))
        inflation_rate = float(args.get("inflation_rate", 0))
        gdp_growth = float(args.get("gdp_growth", 0))

        # Prepare input for prediction
        input_data = pd.DataFrame({
            "interest_rate": [interest_rate],
            "inflation_rate": [inflation_rate],
            "gdp_growth": [gdp_growth]
        })

        # Debugging: Print input data
        print("Input Data:", input_data)

        # Make prediction
        result = model.predict(input_data)

        # Return formatted response
        return {"loan_default_risk": float(result[0])}
    
    except Exception as e:
        return {"error": str(e)}
