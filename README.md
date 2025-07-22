# MeNow-smell_pred
# Fine-Tuning **FART** (Flavor Analysis and Recognition Transformer) to predict Odour Labels

*A multi‑label classification workflow for “pungent”, “sweet”, “floral”, and “minty” scents using the pyrfume-data repository.*

---

## 1. Goal

Given molecular structures from the **pyrfume-data** repository, predict whether each molecule expresses one or more of four odor qualities: **pungent**, **sweet**, **floral**, **minty**. We keep the task open-ended but reproducible and well engineered. The flagship model here is **FART** (a transformer previously used for flavor prediction). We’ll also build simple baselines for sanity checks.

---

## 2. High-Level Workflow

1. **Data intake & unification**

   * Pull relevant CSVs directly from `pyrfume-data` (no need to install the package).
   * Canonicalize molecules via SMILES and merge odor descriptors across datasets.
2. **Label engineering**

   * Map free-text odor descriptors to the four target labels via a synonym/regex dictionary.
   * Produce a clean multi-label target matrix (N molecules × 4 labels).
3. **Feature construction**

   * Compute molecular fingerprints & descriptors (RDKit).
   * Optionally generate learned embeddings (e.g., FART embeddings, ChemBERTa) for comparison.
4. **Modeling**

   * **Baseline**: Logistic Regression / XGBoost on fingerprints.
   * **Main**: Fine-tune or train **FART** for odor label prediction, transforming into SART (Smell Analysis and Recognition Transformer).
5. **Evaluation & analysis**

   * Use per-label PR-AUC, ROC-AUC, F1 (macro/micro).
   * Perform error analysis and interpretability (e.g., SHAP or attention visualization).
6. **Packaging & reporting**

   * Provide notebooks/scripts, config files, and a concise results summary.

---

## 3. Repository Structure

TODO

---

## 4. Data & Label Engineering

TODO

---

## 5. Feature Construction

TODO

---

## 6. Modeling Strategy

### 6.1 Baselines

* **Logistic Regression (One-vs-Rest)** on fingerprints → fast & interpretable weights.
* **XGBoost** with class weights or focal loss for imbalance.

### 6.2 Main Model: **FART**

* **Input**: SMILES strings → tokenizer → transformer encoder.
* **Head**: Multi-label classification (4 sigmoid outputs).
* **Loss**: Binary cross-entropy (with optional per-label weights).
* **Training**:

  * Optimizer: AdamW
  * LR: start \~1e-4–3e-5 for fine-tuning, use linear warmup + cosine decay.
  * Epochs: small dataset → more epochs but early stop on val PR-AUC.

### 6.3 Hyperparameter tuning

* Keep it light (Optuna or grid search over a few key params).
* Track experiments via a simple CSV or MLflow/W\&B if available.

---

## 7. Evaluation

* **Per-label**: Precision-Recall AUC (preferred with imbalance), ROC-AUC, F1-score.
* **Overall**: Micro/macro F1, subset accuracy (strict all-4 match), but report carefully.
* **Calibration**: Evaluate probability calibration (reliability curves).
* **Interpretability**:

  * Tree models: SHAP to find contributing bits/descriptors.
  * FART: attention maps or gradient-based attribution on SMILES tokens.

---

## 8. Error Analysis & Insights

* Examine molecules misclassified as minty vs. true negatives to discover missing synonyms.
* Cluster embeddings (UMAP) to visualize scent regions.
* Report interesting structural motifs for each scent.

---

## 9. Reproducibility & Environment

* Fix random seeds (NumPy, PyTorch, LightGBM).
* Save splits, label matrix, and trained models under `results/`.
* Dependencies (example):

```yaml
name: scent-env
channels: [conda-forge]
dependencies:
  - python=3.10
  - rdkit
  - pandas
  - scikit-learn
  - lightgbm
  - pytorch
  - transformers  # for FART if on HF
  - optuna
  - jupyterlab
  - matplotlib
```

---

## 10. Quickstart

TODO

---

## 11. What to Show in the Report

* Dataset size, label distribution.
* Baseline vs. FART metrics (table + PR curves).
* Brief discussion: where FART helps (e.g., complex multi-label molecules).
* Examples of molecules and predicted probabilities.

---

## 12. Possible Extensions

* Add more odor classes (expand synonym sets).
* Semi-supervised learning using unlabeled pyrfume molecules.
* Zero-shot classification (text embeddings → odor descriptors).
* Substructure highlighting for user-facing explanations.


