"""
Final summary comparison plot: LIME vs SHAP across all 4 models.
Shows importance rankings side by side for each model.
"""
import pandas as pd
import numpy as np
import joblib
import shap
import lime
import lime.lime_tabular
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

Path("results/comparison").mkdir(parents=True, exist_ok=True)


def get_shap_importance(model_name):
    model = joblib.load(f"models/{model_name}.pkl")
    X_test, _, feature_names = joblib.load("models/test_data.pkl")
    if model_name in ("xgboost", "lightgbm", "random_forest"):
        exp = shap.TreeExplainer(model)
    else:
        exp = shap.KernelExplainer(
            model.predict, shap.sample(X_test, 100)
        )
    sv       = exp.shap_values(X_test)
    mean_abs = np.abs(sv).mean(axis=0)
    total    = mean_abs.sum()
    return pd.Series(mean_abs / total * 100, index=feature_names)


def get_lime_importance(model_name, n_instances=80):
    model = joblib.load(f"models/{model_name}.pkl")
    X_test, _, feature_names = joblib.load("models/test_data.pkl")
    lime_exp = lime.lime_tabular.LimeTabularExplainer(
        training_data         = X_test.values,
        feature_names         = feature_names,
        mode                  = "regression",
        discretize_continuous = True,
        random_state          = 42,
    )
    all_coefs = []
    for i in range(n_instances):
        exp = lime_exp.explain_instance(
            data_row     = X_test.values[i],
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
    total    = mean_abs.sum()
    return (mean_abs / total * 100).round(2)


def run_comparison():
    models = ["xgboost", "lightgbm", "random_forest", "mlp"]
    colors = {"SHAP": "#534AB7", "LIME": "#1D9E75"}

    fig, axes = plt.subplots(
        len(models), 2,
        figsize=(14, 5 * len(models)),
        sharey="row"
    )

    for row, model_name in enumerate(models):
        print(f"\nComparing {model_name}...")

        shap_imp = get_shap_importance(model_name)
        print(f"  SHAP done")
        lime_imp = get_lime_importance(model_name)
        print(f"  LIME done")

        # Normalise both to % for fair comparison
        for col, (method, imp) in enumerate(
            [("SHAP", shap_imp), ("LIME", lime_imp)]
        ):
            ax     = axes[row][col]
            sorted_imp = imp.sort_values()
            bar_colors = [colors[method]
                          if i >= len(sorted_imp) - 3
                          else colors[method] + "66"
                          for i in range(len(sorted_imp))]
            ax.barh(sorted_imp.index, sorted_imp.values,
                    color=bar_colors)
            ax.set_title(f"{model_name} — {method}",
                         fontweight="bold", fontsize=11)
            ax.set_xlabel("% of total importance")
            if col == 0:
                ax.set_ylabel("Feature")

    plt.suptitle(
        "LIME vs SHAP feature importance — all 4 models\n"
        "Top 3 features highlighted per method",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig("results/comparison/full_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved: results/comparison/full_comparison.png")


if __name__ == "__main__":
    run_comparison()