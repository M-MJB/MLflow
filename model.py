import mlflow
import mlflow.sklearn
import argparse
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score



mlflow_server_uri = 'https://mlflow-cloud-server.onrender.com'
mlflow.set_tracking_uri(mlflow_server_uri)
mlflow.set_experiment("california house price estimation")