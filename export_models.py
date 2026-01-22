import mlflow
import joblib
import os

os.makedirs("models", exist_ok=True)

XGB_RUN_ID = "8ce72d9bab764bfbb7f59497215f4a39"
LOGREG_RUN_ID = "9d462f0b6efa4c809adef464c8879b23"

mlflow.set_tracking_uri("file:./mlruns")

xgb = mlflow.sklearn.load_model(f"runs:/{XGB_RUN_ID}/model")
logreg = mlflow.sklearn.load_model(f"runs:/{LOGREG_RUN_ID}/model")

joblib.dump(xgb, "models/xgboost.pkl")
joblib.dump(logreg, "models/logreg.pkl")

print("Models exported successfully.")
