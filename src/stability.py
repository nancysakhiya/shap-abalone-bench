"""
Stability test: retrain the model 20 times with different seeds.
Each time compute LIME and SHAP importance rankings.
Measure: how much does the ranking change? (lower std = more stable)

This is your core research comparison.
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
from xgboost import XGBRegressor
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

Path("results/comparison").mkdir(parents=True, exist_ok=True)
N_TRIALS = 20


def run_stability():
    print("=== Stability Comparison: LIME vs SHAP ===\n")

    df           = pd.read_csv("data/abalone.csv")
    X            = df.drop(columns=["Rings"])
    y            = df["Rings"]
    feature_names = X.columns.tolist()

    shap_ranks = []
    lime_ranks = []

    for seed in range(N_TRIALS):
        print(f"  Trial {seed + 1}/{N_TRIALS}...", end=" ", flush=True)

        X_train, X_test, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        model = XGBRegressor(n_estimators=100, random_state=seed,
                              verbosity=0)
        model.fit(X_train, y_train)

        # ── SHAP ranks ───────────────────────────────────────────────────────
        exp_shap  = shap.TreeExplainer(model)
        sv        = exp_shap.shap_values(X_test)
        shap_mean = np.abs(sv).mean(axis=0)
        shap_rank = pd.Series(shap_mean, index=feature_names)\
                      .rank(ascending=False)
        shap_ranks.append(shap_rank.values)

        # ── LIME ranks ───────────────────────────────────────────────────────
        lime_exp = lime.lime_tabular.LimeTabularExplainer(
            training_data         = X_train.values,
            feature_names         = feature_names,
            mode                  = "regression",
            discretize_continuous = True,
            random_state          = seed,
        )
        lime_coefs = []
        # Use 50 instances for speed inside the loop
        for i in range(50):
            instance = X_test.values[i]
            explanation = lime_exp.explain_instance(
                data_row   = instance,
                predict_fn = model.predict,
                num_features = len(feature_names),
                num_samples  = 1000,
            )
            row = {f: 0.0 for f in feature_names}
            for lime_feat, val in explanation.as_list():
                for f in feature_names:
                    if f.lower() in lime_feat.lower():
                        row[f] = max(row[f], abs(val))
            lime_coefs.append(row)

        lime_mean = pd.DataFrame(lime_coefs).mean()
        lime_rank = lime_mean.rank(ascending=False)
        lime_ranks.append(lime_rank.values)
        print("done")

    # ── Compute rank std ─────────────────────────────────────────────────────
    shap_df  = pd.DataFrame(shap_ranks, columns=feature_names)
    lime_df  = pd.DataFrame(lime_ranks, columns=feature_names)

    shap_std = shap_df.std().round(3)
    lime_std = lime_df.std().round(3)

    print("\nSHAP rank stability (std — lower = more stable):")
    print(shap_std.sort_values().to_string())
    print("\nLIME rank stability (std — lower = more stable):")
    print(lime_std.sort_values().to_string())

    shap_score = round(100 * (1 - shap_std.mean() / (len(feature_names) / 2)), 1)
    lime_score = round(100 * (1 - lime_std.mean() / (len(feature_names) / 2)), 1)
    print(f"\nSHAP stability score: {shap_score}/100")
    print(f"LIME stability score: {lime_score}/100")

    # ── Side-by-side box plot ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    shap_df.boxplot(ax=axes[0])
    axes[0].set_title(f"SHAP rank distribution\nStability score: "
                       f"{shap_score}/100", fontweight="bold")
    axes[0].set_ylabel("Rank (1 = most important)")
    axes[0].set_xlabel("Feature")
    axes[0].tick_params(axis="x", rotation=30)

    lime_df.boxplot(ax=axes[1])
    axes[1].set_title(f"LIME rank distribution\nStability score: "
                       f"{lime_score}/100", fontweight="bold")
    axes[1].set_ylabel("Rank (1 = most important)")
    axes[1].set_xlabel("Feature")
    axes[1].tick_params(axis="x", rotation=30)

    plt.suptitle("Stability comparison: SHAP vs LIME\n"
                 "Tighter boxes = more stable = more trustworthy",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/comparison/stability_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved: results/comparison/stability_comparison.png")

    # ── Bar chart of rank std per feature ────────────────────────────────────
    x      = np.arange(len(feature_names))
    width  = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, shap_std.values, width,
           label="SHAP", color="#534AB7", alpha=0.85)
    ax.bar(x + width/2, lime_std.values, width,
           label="LIME", color="#1D9E75", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=30, ha="right")
    ax.set_ylabel("Rank std (lower = more stable)")
    ax.set_title("SHAP vs LIME — rank stability per feature",
                 fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/comparison/stability_per_feature.png", dpi=150)
    plt.close()
    print("Saved: results/comparison/stability_per_feature.png")

    # Save raw data
    shap_df.to_csv("results/comparison/shap_ranks.csv", index=False)
    lime_df.to_csv("results/comparison/lime_ranks.csv", index=False)

    return shap_score, lime_score


if __name__ == "__main__":
    run_stability()