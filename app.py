import gradio as gr
import requests
import json
import os

# CML Model API Endpoint
MODEL_API_URL = "https://modelservice.ml-dbfc64d1-783.go01-dem.ylcu-atmi.cloudera.site/model"
ACCESS_KEY = "mda0gi5d9rezbk3wu23oz9vsewgse2yn"  # Replace if necessary

# Define prediction function
def predict_loan_default(api_key, interest_rate, inflation_rate, gdp_growth):
    payload = {
        "accessKey": ACCESS_KEY,
        "request": {
            "parameters": {
                "interest_rate": interest_rate,
                "inflation_rate": inflation_rate,
                "gdp_growth": gdp_growth
            }
        }
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    response = requests.post(MODEL_API_URL, data=json.dumps(payload), headers=headers)
    if response.status_code == 200:
        result = response.json()
        if result.get("success"):
            return f"Predicted Loan Default Risk: {result['response']['loan_default_risk']:.4f}"
        else:
            return "Error: Model did not return a valid response"
    else:
        return "Error: Unable to fetch prediction"

# Create Gradio Interface
tabbed = gr.Interface(
    fn=predict_loan_default,
    inputs=[
        gr.Textbox(label="API Key", type="password"),
        gr.Number(label="Interest Rate (%)"),
        gr.Number(label="Inflation Rate (%)"),
        gr.Number(label="GDP Growth (%)")
    ],
    outputs=gr.Textbox(label="Loan Default Risk Prediction"),
    title="Loan Default Risk Predictor",
    description="Enter your API key and economic indicators to predict loan default risk using a deployed model."
)

# Run Gradio app
if __name__ == "__main__":
    tabbed.launch(share=True,
                  show_error=True,
                  server_name='127.0.0.1',
                  server_port=int(os.getenv('CDSW_APP_PORT')))