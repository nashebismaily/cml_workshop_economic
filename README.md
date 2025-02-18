# **Economic Data Prediction and Analysis**

This repository contains a machine learning project for analyzing and predicting economic data using **MLflow**, **scikit-learn**, and **Gradio** for model deployment. The project is designed for use in Cloudera's **CDSW (Cloudera Data Science Workbench)** or other environments.

## **Project Structure**
```
├── analysis.ipynb          # Exploratory data analysis and visualization
├── economic_data.csv       # Dataset with economic indicators
├── fit.ipynb               # Model training and evaluation
├── mlflow_experiment.ipynb # MLflow experiment tracking
├── predict.py              # Script for model inference
├── app.py                  # Gradio-based application for model deployment
├── requirements.txt        # Dependencies
├── cdsw-build.sh           # Script for setting up dependencies
```

## **Installation**
Ensure you have Python installed and run:

```bash
pip install -r requirements.txt
```

## **Usage**
### **1. Data Analysis**
Run `analysis.ipynb` to explore the dataset (`economic_data.csv`).

### **2. Model Training**
Execute `fit.ipynb` to train and evaluate the machine learning model.

### **3. Experiment Tracking**
Use `mlflow_experiment.ipynb` to track model performance using MLflow.

### **4. Model Prediction**
Run `predict.py` for making predictions using the trained model.

### **5. Web Application**
Launch the Gradio-based app using:

```bash
python app.py
```
This will start a web-based UI to interact with the model.

## **Dependencies**
The project relies on:
- **pandas** - Data manipulation
- **seaborn** & **matplotlib** - Data visualization
- **scikit-learn** - Machine learning models
- **Gradio** - Web-based UI for ML model
- **MLflow** - Model tracking and experiment logging
