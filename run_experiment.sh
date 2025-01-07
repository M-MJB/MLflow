#!/bin/bash

# Variables
REPO_URL="https://github.com/M-MJB/MLflow.git"  
MLFLOW_TRACKING_URI="https://mlflow-cloud-server.onrender.com/" #"http://127.0.0.1:5000/"
MLFLOW_EXPERIMENT_NAME="experiment 1"
PARAMS="-P lr=0.01 -P test_size=0.2 -P epochs=100" 

# Clone the repository
echo "Cloning the GitHub repository..."
git clone $REPO_URL
cd MLflow

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Set the MLflow tracking URI and experiment name
echo "Setting MLflow tracking URI to $MLFLOW_TRACKING_URI..."
export MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI
export MLFLOW_EXPERIMENT_NAME=$MLFLOW_EXPERIMENT_NAME

# Run the MLflow experiment
echo "Running the MLflow experiment with parameters: $PARAMS..."
mlflow run . $PARAMS

# Completion message
echo "Experiment completed! Check the MLflow UI at $MLFLOW_TRACKING_URI to view results."
