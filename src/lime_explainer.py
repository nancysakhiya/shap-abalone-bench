"""
LIME — Local Interpretable Model-agnostic Explanations

How LIME works in simple words:
  1. Pick one data point (one abalone) you want to explain.
  2. Create hundreds of fake neighbours around it by randomly
     changing its feature values slightly.
  3. Ask the black-box model to predict all those fake neighbours.
  4. Fit a simple linear regression on those fake neighbours,
     weighted by how close they are to the original point.
  5. The coefficients of that linear model are your explanation.
     Positive coefficient = feature pushed prediction UP.
     Negative coefficient = feature pushed prediction DOWN.

Key difference from SHAP:
  SHAP explains globally using game theory (average over all data).
  LIME explains locally using a linear approximation (one point at a time).
"""

import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

Path("results/lime_plots").mkdir(parents=True, exist_ok=True)


def build_lime_explainer(X_train: np.ndarray, feature_names: list):
    """
    Create a LIME explainer for tabular data.
    X_train is used to learn the distribution of each feature
    so LIME knows how to generate realistic neighbours.
    """
    return lime.lime_tabular.LimeTabularExplainer(
        training_data   = X_train,
        feature_names   = feature_names,
        mode            = "regression",   # we are predicting rings (a number)
        discretize_continuous = True,     # LIME bins continuous features
        random_state    = 42,
    )


def explain_single_instance(explainer, model, instance: np.ndarray,
                             feature_names: list, label: str,
                             instance_idx: int):
    """
    Explain one prediction using LIME.
    Returns a dict of {feature_name: coefficient}.
    """
    exp = explainer.explain_instance(
        data_row       = instance,
        predict_fn     = model.predict,
        num_features   = len(feature_names),  # show all features
        num_samples    = 5000,                # fake neighbours to generate
    )
    return dict(exp.as_list())


def compute_lime_importance(model_name: str, n_instances: int = 100):
    """
    Run LIME on n_instances test points, collect all explanations,
    average the absolute coefficients to get global importance.

    Why 100 instances?
    LIME is slow (fits a new model per instance). 100 gives a
    stable average without taking too long.
    """
    model = joblib.load(f"models/{model_name}.pkl")
    X_test, y_test, feature_names = joblib.load("models/test_data.pkl")

    X_train_arr = X_test.values  # use test set distribution for explainer
    explainer   = build_lime_explainer(X_train_arr, feature_names)

    print(f"  Running LIME on {n_instances} instances for {model_name}...")
    all_coefs = []

    for i in range(n_instances):
        instance = X_test.values[i]
        coef_dict = explain_single_instance(
            explainer, model, instance,
            feature_names, model_name, i
        )
        # Take absolute value — we want importance magnitude
        row = {feat: abs(coef_dict.get(feat, 0.0))
               for feat in feature_names}
        # Handle LIME's discretized feature names like "Shell_weight > 0.23"
        # by falling back to partial name matching
        if all(v == 0.0 for v in row.values()):
            for feat in feature_names:
                for lime_feat, val in coef_dict.items():
                    if feat.lower() in lime_feat.lower():
                        row[feat] = max(row[feat], abs(val))
        all_coefs.append(row)

    coef_df   = pd.DataFrame(all_coefs)
    mean_abs  = coef_df.mean()
    total     = mean_abs.sum()
    pct_share = (mean_abs / total * 100).round(2)

    # ── Bar plot of global importance ────────────────────────────────────────
    sorted_idx  = mean_abs.sort_values(ascending=True).index
    sorted_vals = mean_abs[sorted_idx]

    plt.figure(figsize=(8, 5))
    colors = ["#534AB7" if i >= len(sorted_idx) - 3 else "#AFA9EC"
              for i in range(len(sorted_idx))]
    plt.barh(sorted_idx, sorted_vals, color=colors)
    plt.xlabel("Mean |LIME coefficient|", fontsize=12)
    plt.title(f"LIME global feature importance — {model_name}\n"
              f"(averaged over {n_instances} instances)", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"results/lime_plots/{model_name}_importance.png", dpi=150)
    plt.close()

    print(f"  Saved: results/lime_plots/{model_name}_importance.png")
    return mean_abs, pct_share, coef_df


def explain_three_instances(model_name: str):
    """
    Show detailed LIME explanation for 3 specific instances:
      - a young abalone  (rings < 7)
      - a medium abalone (rings 9-11)
      - an old abalone   (rings > 15)
    This gives a sense of how LIME varies locally.
    """
    model = joblib.load(f"models/{model_name}.pkl")
    X_test, y_test, feature_names = joblib.load("models/test_data.pkl")

    y_test_arr = y_test.values
    young_idx  = np.where(y_test_arr < 7)[0][0]
    medium_idx = np.where((y_test_arr >= 9) & (y_test_arr <= 11))[0][0]
    old_idx    = np.where(y_test_arr > 15)[0][0]

    cases = {
        "young (rings<7)":   young_idx,
        "medium (rings9-11)": medium_idx,
        "old (rings>15)":    old_idx,
    }

    explainer = build_lime_explainer(X_test.values, feature_names)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (label, idx) in zip(axes, cases.items()):
        instance = X_test.values[idx]
        pred     = model.predict(instance.reshape(1, -1))[0]
        actual   = y_test_arr[idx]

        exp = explainer.explain_instance(
            data_row   = instance,
            predict_fn = model.predict,
            num_features = len(feature_names),
            num_samples  = 3000,
        )
        items = exp.as_list()
        # Sort by absolute value
        items_sorted = sorted(items, key=lambda x: abs(x[1]), reverse=True)[:6]
        feats  = [x[0].split(">")[0].split("<")[0].split("=")[0].strip()
                  for x in items_sorted]
        vals   = [x[1] for x in items_sorted]
        colors = ["#0F6E56" if v > 0 else "#993C1D" for v in vals]

        ax.barh(feats[::-1], vals[::-1], color=colors[::-1])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{label}\nPredicted: {pred:.1f}  Actual: {actual}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("LIME coefficient", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)

    plt.suptitle(f"LIME local explanations — {model_name}\n"
                 f"Green = pushes prediction UP, Red = pushes DOWN",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"results/lime_plots/{model_name}_local_3cases.png", dpi=150)
    plt.close()
    print(f"  Saved: results/lime_plots/{model_name}_local_3cases.png")


def run_all_lime():
    all_importance = {}
    for name in ("xgboost", "lightgbm", "random_forest", "mlp"):
        print(f"\nLIME — {name}")
        mean_abs, pct, coef_df = compute_lime_importance(name, n_instances=100)
        explain_three_instances(name)
        all_importance[name] = mean_abs
        coef_df.to_csv(f"results/lime_plots/{name}_coefs.csv", index=False)

    print("\nAll LIME analysis complete.")
    return all_importance


if __name__ == "__main__":
    run_all_lime()