import shap
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

Path("results/shap_plots").mkdir(parents=True, exist_ok=True)


def get_explainer(model_name: str, model, X_background):
    if model_name in ("xgboost", "lightgbm", "random_forest"):
        return shap.TreeExplainer(model)
    elif model_name == "mlp":
        return shap.KernelExplainer(
            model.predict, shap.sample(X_background, 100)
        )
    raise ValueError(f"Unknown model: {model_name}")


def compute_shap_importance(model_name: str):
    model = joblib.load(f"models/{model_name}.pkl")
    X_test, y_test, feature_names = joblib.load("models/test_data.pkl")

    print(f"  Computing SHAP for {model_name}...")
    explainer  = get_explainer(model_name, model, X_test)
    shap_vals  = explainer.shap_values(X_test)
    mean_abs   = np.abs(shap_vals).mean(axis=0)
    total      = mean_abs.sum()
    pct_share  = (mean_abs / total * 100).round(2)

    np.save(f"results/shap_plots/{model_name}_shap_values.npy", shap_vals)

    # Bar plot
    import pandas as pd
    s = pd.Series(mean_abs, index=feature_names).sort_values()
    plt.figure(figsize=(8, 5))
    colors = ["#534AB7" if i >= len(s) - 3 else "#AFA9EC"
              for i in range(len(s))]
    plt.barh(s.index, s.values, color=colors)
    plt.xlabel("Mean |SHAP value|", fontsize=12)
    plt.title(f"SHAP global feature importance — {model_name}", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"results/shap_plots/{model_name}_importance.png", dpi=150)
    plt.close()

    # Beeswarm
    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_vals, X_test, feature_names=feature_names,
                      show=False)
    plt.title(f"SHAP beeswarm — {model_name}")
    plt.tight_layout()
    plt.savefig(f"results/shap_plots/{model_name}_beeswarm.png", dpi=150)
    plt.close()

    print(f"  Saved plots for {model_name}")
    return pd.Series(mean_abs, index=feature_names)


def run_all_shap():
    import pandas as pd
    all_importance = {}
    for name in ("xgboost", "lightgbm", "random_forest", "mlp"):
        print(f"\nSHAP — {name}")
        all_importance[name] = compute_shap_importance(name)
    print("\nAll SHAP analysis complete.")
    return all_importance


if __name__ == "__main__":
    run_all_shap()