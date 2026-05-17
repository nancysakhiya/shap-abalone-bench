import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

Path("results/shap_plots").mkdir(parents=True, exist_ok=True)


def get_explainer(model_name: str, model, X_background):
    """Returns the right SHAP explainer type per model."""
    if model_name in ("xgboost", "lightgbm", "random_forest"):
        return shap.TreeExplainer(model)
    elif model_name == "mlp":
        return shap.KernelExplainer(model.predict, shap.sample(X_background, 100))
    else:
        raise ValueError(f"Unknown model: {model_name}")


def compute_and_plot_shap(model_name: str):
    model = joblib.load(f"models/{model_name}.pkl")
    X_test, y_test, feature_names = joblib.load("models/test_data.pkl")

    print(f"\nComputing SHAP values for {model_name}...")
    explainer = get_explainer(model_name, model, X_test)
    shap_values = explainer.shap_values(X_test)

    # Save raw values for later stability analysis
    np.save(f"results/shap_plots/{model_name}_shap_values.npy", shap_values)

    # Summary bar plot (global importance)
    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                      plot_type="bar", show=False)
    plt.title(f"SHAP feature importance — {model_name}")
    plt.tight_layout()
    plt.savefig(f"results/shap_plots/{model_name}_bar.png", dpi=150)
    plt.close()

    # Beeswarm plot (direction of impact)
    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title(f"SHAP beeswarm — {model_name}")
    plt.tight_layout()
    plt.savefig(f"results/shap_plots/{model_name}_beeswarm.png", dpi=150)
    plt.close()

    print(f"  Plots saved for {model_name}")
    return shap_values


def run_all_shap():
    for name in ("xgboost", "lightgbm", "random_forest", "mlp"):
        compute_and_plot_shap(name)
    print("\nAll SHAP plots saved to results/shap_plots/")


if __name__ == "__main__":
    run_all_shap()