import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

Path("results").mkdir(exist_ok=True)


# ── Experiment 1: Stability ─────────────────────────────────────────────────
# Re-run SHAP 30 times with different random seeds.
# Measure: how much does the feature ranking change?

def stability_experiment(n_trials: int = 30):
    print("\n=== Experiment 1: SHAP Stability ===")
    df = pd.read_csv("data/abalone.csv")
    X = df.drop(columns=["Rings"])
    y = df["Rings"]
    feature_names = X.columns.tolist()

    rank_matrix = []  # shape: (n_trials, n_features)

    for seed in range(n_trials):
        X_train, X_test, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        model = XGBRegressor(n_estimators=100, random_state=seed, verbosity=0)
        model.fit(X_train, y_train)

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_test)
        mean_abs = np.abs(sv).mean(axis=0)

        # Rank features: rank 1 = most important
        ranks = pd.Series(mean_abs, index=feature_names).rank(ascending=False)
        rank_matrix.append(ranks.values)

    rank_df = pd.DataFrame(rank_matrix, columns=feature_names)

    # Stability score: lower std = more stable
    stability = rank_df.std()
    print("\nFeature rank stability (std of rank across trials, lower = more stable):")
    print(stability.sort_values())

    # Plot
    fig, ax = plt.subplots(figsize=(9, 4))
    rank_df.boxplot(ax=ax, vert=True)
    ax.set_title("SHAP feature rank distribution across 30 random seeds")
    ax.set_ylabel("Rank (1 = most important)")
    ax.set_xlabel("Feature")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("results/stability_boxplot.png", dpi=150)
    plt.close()

    rank_df.to_csv("results/stability_ranks.csv", index=False)
    print("Saved: results/stability_boxplot.png, results/stability_ranks.csv")
    return stability


# ── Experiment 2: Faithfulness ───────────────────────────────────────────────
# Mask top-k SHAP features and see how much model performance drops.
# If SHAP found the right features, performance should drop sharply.

def faithfulness_experiment():
    print("\n=== Experiment 2: Faithfulness Test ===")
    df = pd.read_csv("data/abalone.csv")
    X = df.drop(columns=["Rings"])
    y = df["Rings"]
    feature_names = X.columns.tolist()
    n_features = len(feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_test)
    mean_abs = np.abs(sv).mean(axis=0)
    importance_order = np.argsort(mean_abs)[::-1]  # most → least important

    rmse_results = []

    for k in range(1, n_features + 1):
        top_k_features = [feature_names[i] for i in importance_order[:k]]
        sub_model = XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
        sub_model.fit(X_train[top_k_features], y_train)
        y_pred = sub_model.predict(X_test[top_k_features])
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_results.append({"k": k, "features": top_k_features, "rmse": rmse})
        print(f"  Top-{k} features: RMSE = {rmse:.4f}")

    result_df = pd.DataFrame(rmse_results)
    result_df.to_csv("results/faithfulness.csv", index=False)

    # Plot
    plt.figure(figsize=(7, 4))
    plt.plot(result_df["k"], result_df["rmse"], marker="o", color="#534AB7")
    plt.xlabel("Number of top SHAP features used")
    plt.ylabel("RMSE")
    plt.title("Faithfulness: RMSE as we add more SHAP-ranked features")
    plt.xticks(range(1, n_features + 1))
    plt.tight_layout()
    plt.savefig("results/faithfulness_curve.png", dpi=150)
    plt.close()
    print("Saved: results/faithfulness_curve.png, results/faithfulness.csv")
    return result_df


if __name__ == "__main__":
    stability_experiment()
    faithfulness_experiment()
    print("\nAll experiments complete. Check results/")