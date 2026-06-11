"""
Faithfulness test for both LIME and SHAP.
Question: if we use only the top-k features ranked by each method,
how quickly does model performance recover?
Steeper curve = method correctly identified important features.
"""
import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

Path("results/comparison").mkdir(parents=True, exist_ok=True)


def faithfulness_shap(X_train, X_test, y_train, y_test, feature_names):
    model = XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    exp      = shap.TreeExplainer(model)
    sv       = exp.shap_values(X_test)
    mean_abs = np.abs(sv).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1]

    results = []
    for k in range(1, len(feature_names) + 1):
        top_k = [feature_names[i] for i in order[:k]]
        m     = XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
        m.fit(X_train[top_k], y_train)
        rmse  = np.sqrt(mean_squared_error(
            y_test, m.predict(X_test[top_k])
        ))
        results.append({"k": k, "rmse": rmse,
                        "feature_added": feature_names[order[k-1]]})
    return pd.DataFrame(results)


def faithfulness_lime(X_train, X_test, y_train, y_test, feature_names):
    model = XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
    model.fit(X_train, y_train)

    # Get LIME importance over 100 instances
    lime_exp = lime.lime_tabular.LimeTabularExplainer(
        training_data         = X_train.values,
        feature_names         = feature_names,
        mode                  = "regression",
        discretize_continuous = True,
        random_state          = 42,
    )
    print("  Computing LIME importance for faithfulness (100 instances)...")
    all_coefs = []
    for i in range(100):
        instance = X_test.values[i]
        exp = lime_exp.explain_instance(
            data_row     = instance,
            predict_fn   = model.predict,
            num_features = len(feature_names),
            num_samples  = 1000,
        )
        row = {f: 0.0 for f in feature_names}
        for lime_feat, val in exp.as_list():
            for f in feature_names:
                if f.lower() in lime_feat.lower():
                    row[f] = max(row[f], abs(val))
        all_coefs.append(row)

    mean_abs = pd.DataFrame(all_coefs).mean()
    order    = mean_abs.sort_values(ascending=False).index.tolist()

    results = []
    for k in range(1, len(feature_names) + 1):
        top_k = order[:k]
        m     = XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
        m.fit(X_train[top_k], y_train)
        rmse  = np.sqrt(mean_squared_error(
            y_test, m.predict(X_test[top_k])
        ))
        results.append({"k": k, "rmse": rmse,
                        "feature_added": order[k-1]})
    return pd.DataFrame(results)


def run_faithfulness():
    print("=== Faithfulness Comparison: LIME vs SHAP ===\n")

    df            = pd.read_csv("data/abalone.csv")
    X             = df.drop(columns=["Rings"])
    y             = df["Rings"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Running SHAP faithfulness...")
    shap_faith = faithfulness_shap(
        X_train, X_test, y_train, y_test, feature_names
    )

    print("Running LIME faithfulness...")
    lime_faith = faithfulness_lime(
        X_train, X_test, y_train, y_test, feature_names
    )

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    shap_faith.to_csv("results/comparison/shap_faithfulness.csv", index=False)
    lime_faith.to_csv("results/comparison/lime_faithfulness.csv", index=False)

    # ── Plot ──────────────────────────────────────────────────────────────────
    full_rmse = shap_faith["rmse"].iloc[-1]

    plt.figure(figsize=(9, 5))
    plt.plot(shap_faith["k"], shap_faith["rmse"],
             marker="o", color="#534AB7", lw=2,
             label="SHAP", markersize=5)
    plt.plot(lime_faith["k"], lime_faith["rmse"],
             marker="s", color="#1D9E75", lw=2,
             label="LIME", markersize=5, linestyle="--")
    plt.axhline(full_rmse, color="gray", linestyle=":",
                lw=1, label=f"Full model RMSE ({full_rmse:.3f})")
    plt.xlabel("Number of top-ranked features used", fontsize=12)
    plt.ylabel("RMSE", fontsize=12)
    plt.title("Faithfulness: SHAP vs LIME\n"
              "Lower RMSE with fewer features = better explanation",
              fontsize=11, fontweight="bold")
    plt.xticks(range(1, len(feature_names) + 1))
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("results/comparison/faithfulness_comparison.png", dpi=150)
    plt.close()
    print("\nSaved: results/comparison/faithfulness_comparison.png")

    print("\nSHAP faithfulness curve:")
    print(shap_faith[["k", "feature_added", "rmse"]].to_string(index=False))
    print("\nLIME faithfulness curve:")
    print(lime_faith[["k", "feature_added", "rmse"]].to_string(index=False))

    return shap_faith, lime_faith


if __name__ == "__main__":
    run_faithfulness()