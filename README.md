# SHAP Abalone Bench (LIME comparison branch)

This branch extends the base SHAP benchmark with a second explanation method, LIME (Local Interpretable Model-agnostic Explanations), and runs both methods head to head on the same models and the same two research questions:

1. Stability — does the feature ranking stay consistent when the model is retrained with a different random seed?
2. Faithfulness — if the method says a feature is important, does removing it actually hurt model performance?

The goal is not just to explain the models, but to find out which explanation method can actually be trusted.

## What is different from the main branch

The main branch only evaluates SHAP. This branch adds:

- A full LIME explainer pipeline (src/lime_explainer.py), producing global importance and local per-instance explanations
- A side-by-side stability comparison of SHAP vs LIME rankings across 20 retrains (src/stability.py)
- A side-by-side faithfulness comparison of SHAP vs LIME (src/faithfulness.py)
- A combined importance comparison plot across all four models (src/compare.py)
- mlflow experiment tracking has been removed from train_models.py on this branch; training runs locally without a tracking server

## Dataset

The UCI Abalone dataset (https://archive.ics.uci.edu/dataset/1/abalone) is used throughout. The task is to predict Rings (a proxy for age) from physical measurements of the abalone.

Feature: Sex — Encoded as M=0, F=1, I=2
Feature: Length — Longest shell measurement
Feature: Diameter — Perpendicular to length
Feature: Height — Height with meat in shell
Feature: Whole_weight — Whole abalone weight
Feature: Shucked_weight — Weight of the meat
Feature: Viscera_weight — Gut weight after bleeding
Feature: Shell_weight — Weight after being dried
Feature: Rings — Target variable (age in years is approximately rings + 1.5)

Data is fetched via ucimlrepo and cached locally as data/abalone.csv.

## Models

The same four regressors as the main branch, trained on an 80/20 split (features standardized for the MLP):

Model: XGBoost — Library: xgboost — 200 estimators, max depth 5
Model: LightGBM — Library: lightgbm — 200 estimators, max depth 5
Model: Random Forest — Library: scikit-learn — 200 estimators, max depth 10
Model: MLP — Library: scikit-learn — Two hidden layers (128, 64), early stopping

## How LIME works, briefly

LIME explains one prediction at a time rather than the model as a whole:

1. Pick a single data point to explain
2. Generate hundreds of synthetic neighbors by slightly perturbing its feature values
3. Get the black-box model's predictions for all of those neighbors
4. Fit a simple weighted linear regression on the neighbors, weighted by distance to the original point
5. The coefficients of that local linear model become the explanation — a positive coefficient pushes the prediction up, a negative one pushes it down

This is the key difference from SHAP: SHAP is grounded in cooperative game theory and produces a globally consistent attribution, while LIME is a local linear approximation that can vary from one point to the next.

## Methodology

### Global SHAP and LIME explanations (per model)

For each of the four trained models, both methods produce a global feature importance plot:

- SHAP: results/shap_plots/{model}_importance.png and results/shap_plots/{model}_beeswarm.png
- LIME: results/lime_plots/{model}_importance.png, averaged over 100 test instances

LIME also produces detailed local explanations for three representative cases per model — a young abalone (rings < 7), a medium one (rings 9-11), and an old one (rings > 15):

- results/lime_plots/{model}_local_3cases.png

Raw values are saved alongside the plots: results/shap_plots/{model}_shap_values.npy and results/lime_plots/{model}_coefs.csv.

### Experiment 1: Stability comparison

An XGBoost model is retrained 20 times with different seeds. On each run, both SHAP and LIME rank the features by importance, and the standard deviation of each feature's rank across the 20 runs is measured — lower means the method agrees with itself more often, regardless of random seed.

On this run, SHAP was noticeably more stable than LIME:

Method: SHAP — Stability score: 86.5 / 100
Method: LIME — Stability score: 77.8 / 100

Per-feature rank standard deviation (lower is more stable):

Feature: Shucked_weight — SHAP std: 0.000 — LIME std: 0.571
Feature: Shell_weight — SHAP std: 0.000 — LIME std: 0.761
Feature: Whole_weight — SHAP std: 0.000 — LIME std: 0.616
Feature: Diameter — SHAP std: 1.119 — LIME std: 0.716
Feature: Length — SHAP std: 1.142 — LIME std: 1.293
Feature: Height — SHAP std: 0.821 — LIME std: 0.933
Feature: Viscera_weight — SHAP std: 0.887 — LIME std: 1.373
Feature: Sex — SHAP std: 0.366 — LIME std: 0.826

SHAP was perfectly consistent on the three weight-related features across every retrain, while LIME's rankings shifted more from run to run, likely a byproduct of its perturbation-based sampling.

Output: results/comparison/stability_comparison.png, results/comparison/stability_per_feature.png, results/comparison/shap_ranks.csv, results/comparison/lime_ranks.csv

### Experiment 2: Faithfulness comparison

Starting from each method's single most important feature, features are added one at a time in ranked order and a fresh XGBoost model is retrained on each subset. A faithful ranking should show RMSE drop quickly as the top few features are added.

Top-1 — SHAP RMSE: 2.508 (Shell_weight) — LIME RMSE: 2.906 (Shucked_weight)
Top-2 — SHAP RMSE: 2.587 (Shucked_weight) — LIME RMSE: 2.582 (Shell_weight)
Top-3 — SHAP RMSE: 2.542 (Whole_weight) — LIME RMSE: 2.543 (Whole_weight)
Top-4 — SHAP RMSE: 2.445 (Diameter) — LIME RMSE: 2.440 (Diameter)
Top-5 — SHAP RMSE: 2.425 (Length) — LIME RMSE: 2.398 (Viscera_weight)
Top-6 — SHAP RMSE: 2.404 (Viscera_weight) — LIME RMSE: 2.431 (Length)
Top-7 — SHAP RMSE: 2.449 (Height) — LIME RMSE: 2.450 (Height)
Top-8 — SHAP RMSE: 2.457 (Sex) — LIME RMSE: 2.459 (Sex)

Both methods converge to nearly the same RMSE once all eight features are included, as expected, but SHAP's single most important feature alone (Shell_weight) gets closer to the full-model RMSE than LIME's single top pick (Shucked_weight), suggesting SHAP's very top-ranked feature is a slightly more faithful summary of what actually drives the model on this dataset.

Output: results/comparison/faithfulness_comparison.png, results/comparison/shap_faithfulness.csv, results/comparison/lime_faithfulness.csv

### Combined comparison

src/compare.py produces one figure showing normalized (percent of total) importance for both methods across all four models side by side, with each method's top three features highlighted.

Output: results/comparison/full_comparison.png

## Project structure

shap-abalone-bench (lime-abalone-branch)/

data/
  abalone.csv                       processed dataset used by all scripts

models/
  xgboost.pkl
  lightgbm.pkl
  random_forest.pkl
  mlp.pkl
  scaler.pkl                        StandardScaler fit on training data
  test_data.pkl                     held-out X_test, y_test, feature names

results/
  shap_plots/
    {model}_importance.png          SHAP global bar plot per model
    {model}_beeswarm.png            SHAP beeswarm plot per model
    {model}_shap_values.npy         raw SHAP values per model
  lime_plots/
    {model}_importance.png          LIME global bar plot per model
    {model}_local_3cases.png        LIME local explanations for 3 sample cases
    {model}_coefs.csv               raw LIME coefficients per model
  comparison/
    stability_comparison.png        SHAP vs LIME rank boxplots
    stability_per_feature.png       SHAP vs LIME rank std bar chart
    shap_ranks.csv                  raw SHAP ranks across 20 seeds
    lime_ranks.csv                  raw LIME ranks across 20 seeds
    faithfulness_comparison.png     SHAP vs LIME RMSE curve
    shap_faithfulness.csv           raw SHAP faithfulness data
    lime_faithfulness.csv           raw LIME faithfulness data
    full_comparison.png             combined importance comparison, all models

src/
  data_loader.py                    fetches and encodes the Abalone dataset
  train_models.py                   trains all four models (no mlflow on this branch)
  shap_explainer.py                 computes and plots SHAP values per model
  lime_explainer.py                 computes and plots LIME explanations per model
  stability.py                      SHAP vs LIME stability comparison (20 seeds)
  faithfulness.py                   SHAP vs LIME faithfulness comparison
  compare.py                        combined importance comparison plot

requirements.txt

## Installation

git clone -b lime-abalone-branch https://github.com/nancysakhiya/shap-abalone-bench.git
cd shap-abalone-bench
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Core libraries: shap, lime, xgboost, lightgbm, scikit-learn, scikit-image, pandas, numpy, matplotlib, seaborn, joblib, ucimlrepo.

## Usage

Run each stage from the repository root, in order.

1. Fetch and prepare the dataset

python src/data_loader.py

2. Train all models

python src/train_models.py

Trains XGBoost, LightGBM, Random Forest, and MLP, prints RMSE and R2 for each, and saves models plus the held-out test split to models/.

3. Generate SHAP explanations

python src/shap_explainer.py

4. Generate LIME explanations

python src/lime_explainer.py

Note: LIME fits a fresh local model per instance, so this step is noticeably slower than the SHAP step (100 instances per model type).

5. Run the stability comparison (SHAP vs LIME, 20 seeds)

python src/stability.py

6. Run the faithfulness comparison (SHAP vs LIME)

python src/faithfulness.py

7. Generate the combined comparison plot

python src/compare.py

## Key finding

On this dataset and these models, SHAP produced more stable feature rankings across retrains than LIME (86.5 vs 77.8 on the stability score used here), and its top-ranked feature alone came slightly closer to the full model's performance in the faithfulness test. Both methods converge to similar conclusions once most features are included, and both agree that Shell_weight, Shucked_weight, and Whole_weight dominate the prediction, but SHAP appears more self-consistent when the model is retrained.
