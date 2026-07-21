# SHAP Abalone Bench

A small benchmark for evaluating the stability and faithfulness of SHAP explanations across different regression models, using the UCI Abalone dataset as a testbed.

Explainability methods like SHAP are widely trusted to describe why a model made a prediction, but the explanations themselves are rarely audited. This project asks two practical questions:

1. Stability — If I retrain the same model type with a different random seed, does SHAP agree on which features matter most?
2. Faithfulness — If SHAP says a feature is important, does removing it actually hurt the model's performance?

## Dataset

The UCI Abalone dataset (https://archive.ics.uci.edu/dataset/1/abalone) is used throughout. The task is to predict the number of Rings (a proxy for age) from physical measurements of the abalone.

| Feature | Description |
|---|---|
| Sex | Encoded as M=0, F=1, I=2 |
| Length | Longest shell measurement |
| Diameter | Perpendicular to length |
| Height | Height with meat in shell |
| Whole_weight | Whole abalone weight |
| Shucked_weight | Weight of the meat |
| Viscera_weight | Gut weight after bleeding |
| Shell_weight | Weight after being dried |
| Rings | Target variable (age in years ≈ rings + 1.5) |

Data is fetched programmatically from the UCI repository via ucimlrepo and cached locally as data/abalone.csv.

## Models

Four regressors are trained on an 80/20 train-test split (features standardized for the MLP):

| Model | Library | Notes |
|---|---|---|
| XGBoost | xgboost | 200 estimators, max depth 5 |
| LightGBM | lightgbm | 200 estimators, max depth 5 |
| Random Forest | scikit-learn | 200 estimators, max depth 10 |
| MLP | scikit-learn | Two hidden layers (128, 64), early stopping |

Training runs are logged with mlflow under the experiment name shap-abalone-bench, and trained models are pickled to models/.

## Methodology

### Experiment 1: Stability

An XGBoost model is retrained 30 times, each with a different random seed and train/test split. For each run, SHAP values are computed on the test set and features are ranked by mean absolute SHAP value. The standard deviation of each feature's rank across the 30 runs is used as a stability score — lower means the feature's importance ranking is more consistent regardless of random seed.

Output: results/stability_boxplot.png, results/stability_ranks.csv

### Experiment 2: Faithfulness

A single XGBoost model is trained and explained with SHAP, giving a global feature importance ranking. Starting from the single most important feature, additional features are added one at a time (in SHAP-ranked order) and a fresh model is retrained on each subset. RMSE is tracked as more "important" features are added — if SHAP's ranking is faithful to the model's actual behavior, RMSE should improve quickly as the top-ranked features are included.

Output: results/faithfulness_curve.png, results/faithfulness.csv

### Per-model SHAP plots

For each of the four trained models, both a global bar plot and a beeswarm plot are generated to visualize feature importance and the direction of each feature's effect. Tree-based models (XGBoost, LightGBM, Random Forest) use shap.TreeExplainer; the MLP uses shap.KernelExplainer with a 100-sample background set, since it has no built-in Shapley approximation.

Output: results/shap_plots/{model}_bar.png, results/shap_plots/{model}_beeswarm.png, and raw values in results/shap_plots/{model}_shap_values.npy

## Project structure

shap-abalone-bench/
├── data/
│   ├── abalone.csv              # processed dataset used by all scripts
│   └── abalone/                 # raw UCI files (data, names, index)
├── models/
│   ├── xgboost.pkl
│   ├── lightgbm.pkl
│   ├── random_forest.pkl
│   ├── mlp.pkl
│   ├── scaler.pkl                # StandardScaler fit on training data
│   └── test_data.pkl              # held-out X_test, y_test, feature names
├── results/
│   ├── stability_boxplot.png
│   ├── stability_ranks.csv
│   ├── faithfulness_curve.png
│   ├── faithfulness.csv
│   └── shap_plots/                # per-model bar/beeswarm plots + raw SHAP arrays
├── src/
│   ├── data_loader.py             # fetches and encodes the Abalone dataset
│   ├── train_models.py            # trains all four models, logs to mlflow
│   ├── shap_explainer.py          # computes and plots SHAP values per model
│   └── experiments.py             # stability and faithfulness experiments
└── requirements.txt

## Installation

git clone https://github.com/nancysakhiya/shap-abalone-bench.git
cd shap-abalone-bench
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt

The core libraries used are shap, xgboost, lightgbm, scikit-learn, pandas, numpy, matplotlib, mlflow, joblib, and ucimlrepo. If any of these are missing from your environment, install them directly:

pip install shap xgboost lightgbm scikit-learn pandas numpy matplotlib mlflow joblib ucimlrepo

## Usage

Run each stage from the repository root, in order:

1. Fetch and prepare the dataset

python src/data_loader.py

2. Train all models

python src/train_models.py

This trains XGBoost, LightGBM, Random Forest, and MLP, prints RMSE and R² for each, and saves models plus the held-out test split to models/.

3. Generate SHAP explanations

python src/shap_explainer.py

This produces bar and beeswarm plots, and saves raw SHAP value arrays, for every trained model.

4. Run the stability and faithfulness experiments

python src/experiments.py

This runs both experiments end to end and writes all outputs to results/.

To inspect training runs and metrics with MLflow's UI:

mlflow ui

## Sample results

From a full run of the faithfulness experiment on XGBoost, RMSE by number of top SHAP-ranked features included:

| Top-k features | RMSE |
|---|---|
| 1 (Shell_weight) | 2.508 |
| 2 (+ Shucked_weight) | 2.587 |
| 3 (+ Whole_weight) | 2.542 |
| 4 (+ Diameter) | 2.445 |
| 5 (+ Length) | 2.425 |
| 6 (+ Viscera_weight) | 2.404 |
| 7 (+ Height) | 2.449 |
| 8 (+ Sex) | 2.457 |

Shell_weight alone captures most of the predictive signal, and RMSE bottoms out around the top six features, roughly consistent with SHAP's ranking of feature importance.
## License

No license file is currently included in this repository. Add one (for example MIT or Apache 2.0) if you intend for others to reuse or contribute to this code.
