import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from pathlib import Path
import joblib

MODELS = {
    "xgboost": XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                             random_state=42, verbosity=0),
    "lightgbm": LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                               random_state=42, verbose=-1),
    "random_forest": RandomForestRegressor(n_estimators=200, max_depth=10,
                                            random_state=42, n_jobs=-1),
    "mlp": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500,
                        random_state=42, early_stopping=True),
}


def prepare_data(csv_path: str = "data/abalone.csv"):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["Rings"])
    y = df["Rings"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale for MLP — tree models don't need it but it doesn't hurt
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()


def train_all(csv_path: str = "data/abalone.csv"):
    Path("models").mkdir(exist_ok=True)
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(csv_path)

    mlflow.set_experiment("shap-abalone-bench")
    results = {}

    for name, model in MODELS.items():
        with mlflow.start_run(run_name=name):
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            mlflow.log_param("model", name)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)

            # Save model to disk
            joblib.dump(model, f"models/{name}.pkl")
            print(f"  RMSE: {rmse:.4f}  |  R²: {r2:.4f}")

            results[name] = {
                "model": model,
                "rmse": rmse,
                "r2": r2,
            }

    # Save test data for SHAP
    joblib.dump((X_test, y_test, feature_names), "models/test_data.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print("\nAll models trained and saved to models/")
    return results


if __name__ == "__main__":
    results = train_all()
    print("\nSummary:")
    for name, r in results.items():
        print(f"  {name:20s}  RMSE={r['rmse']:.4f}  R²={r['r2']:.4f}")