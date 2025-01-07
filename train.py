import mlflow
import mlflow.sklearn
import argparse
import platform
import psutil
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss


# mlflow_server_uri = 'http://127.0.0.1:5000/'
# mlflow.set_tracking_uri(mlflow_server_uri)
# mlflow.set_experiment("exp2")

def train_model(learning_rate, test_size, epochs):
    with mlflow.start_run():
        start_time = time.time()

        # Load dataset
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=test_size, random_state=42)

        #train
        model = RandomForestClassifier(n_estimators=epochs)
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        loss = log_loss(y_test, y_pred_proba)


        #hyper parameters
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("epochs", epochs)

        mlflow.log_param("system", platform.system())
        mlflow.log_param("processor", platform.processor())
        mlflow.log_param("cpu_cores", psutil.cpu_count(logical=False))
        mlflow.log_param("memory_gb", round(psutil.virtual_memory().total / (1024 ** 3), 2))
        mlflow.log_param("python_version", platform.python_version())


        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("loss", loss)

        #model
        mlflow.sklearn.log_model(model, "SGDRegressor_model")
        
        # Log runtime
        runtime_seconds = time.time() - start_time
        mlflow.log_metric("runtime_seconds", runtime_seconds)


if __name__=="__main__":
    #sample : python train.py --lr 0.5 --test_size 0.1 --epochs 100
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for the model")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of test data")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")

    args = parser.parse_args()

    train_model(args.lr, args.test_size, args.epochs )


