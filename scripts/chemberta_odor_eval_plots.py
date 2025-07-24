import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
)

LABELS = ['sweet', 'floral', 'mint', 'pungent']
MODEL_DIR = 'chemberta_lora_results/final_model'
TEST_CSV = 'data/pyrfume_test_4odors.csv'
PLOTS_DIR = 'chemberta_lora_results/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Load model and tokenizer ---
print('Loading model and tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
base_model = AutoModelForSequenceClassification.from_pretrained(
    'seyonec/SMILES_tokenized_PubChem_shard00_160k',
    num_labels=len(LABELS),
    problem_type='multi_label_classification'
)
model = PeftModel.from_pretrained(base_model, MODEL_DIR)
model.eval()

# --- Load test set ---
df = pd.read_csv(TEST_CSV)

# --- Tokenize ---
def tokenize_smiles(smiles_list, tokenizer, max_length=256):
    return tokenizer(smiles_list, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')

inputs = tokenize_smiles(df['SMILES'].tolist(), tokenizer)

# --- Inference ---
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits.cpu().numpy()
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)

y_true = df[LABELS].values

# --- Metrics ---
results = {}
for i, label in enumerate(LABELS):
    y_true_label = y_true[:, i]
    y_pred_label = preds[:, i]
    y_prob_label = probs[:, i]
    if len(np.unique(y_true_label)) > 1:
        precision, recall, f1, _ = precision_recall_fscore_support(y_true_label, y_pred_label, average='binary', zero_division=0)
        auc = roc_auc_score(y_true_label, y_prob_label)
        pr_auc = average_precision_score(y_true_label, y_prob_label)
    else:
        precision = recall = f1 = auc = pr_auc = 0.0
    results[f'{label}_precision'] = precision
    results[f'{label}_recall'] = recall
    results[f'{label}_f1'] = f1
    results[f'{label}_auc'] = auc
    results[f'{label}_pr_auc'] = pr_auc

# --- Plots ---
plt.style.use('default')
sns.set_palette('husl')

# ROC Curves
plt.figure(figsize=(12, 8))
colors = sns.color_palette('husl', len(LABELS))
for i, (label, color) in enumerate(zip(LABELS, colors)):
    if len(np.unique(y_true[:, i])) > 1:
        fpr, tpr, _ = roc_curve(y_true[:, i], probs[:, i])
        auc_score = roc_auc_score(y_true[:, i], probs[:, i])
        plt.plot(fpr, tpr, color=color, linewidth=2, label=f'{label.upper()} (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves by Odor Class', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
roc_path = os.path.join(PLOTS_DIR, 'roc_curves.png')
plt.savefig(roc_path, dpi=300, bbox_inches='tight')
plt.close()
print(f'ROC curves saved to: {roc_path}')

# PR Curves
plt.figure(figsize=(12, 8))
for i, (label, color) in enumerate(zip(LABELS, colors)):
    if len(np.unique(y_true[:, i])) > 1:
        precision, recall, _ = precision_recall_curve(y_true[:, i], probs[:, i])
        pr_auc = average_precision_score(y_true[:, i], probs[:, i])
        plt.plot(recall, precision, color=color, linewidth=2, label=f'{label.upper()} (PR-AUC = {pr_auc:.3f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curves by Odor Class', fontsize=14, fontweight='bold')
plt.legend(loc='lower left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
pr_path = os.path.join(PLOTS_DIR, 'precision_recall_curves.png')
plt.savefig(pr_path, dpi=300, bbox_inches='tight')
plt.close()
print(f'PR curves saved to: {pr_path}')

# Performance Summary
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
class_counts = y_true.sum(axis=0)
bars = plt.bar(LABELS, class_counts, color=colors)
plt.xlabel('Odor Class', fontsize=12)
plt.ylabel('Number of Positive Samples', fontsize=12)
plt.title('Class Distribution in Test Set', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
for bar, count in zip(bars, class_counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{int(count)}', ha='center', va='bottom', fontsize=10)
plt.subplot(1, 2, 2)
auc_scores = [results[f'{label}_auc'] for label in LABELS]
pr_auc_scores = [results[f'{label}_pr_auc'] for label in LABELS]
x_pos = np.arange(len(LABELS))
width = 0.35
plt.bar(x_pos - width/2, auc_scores, width, label='ROC-AUC', color='skyblue')
plt.bar(x_pos + width/2, pr_auc_scores, width, label='PR-AUC', color='lightcoral')
plt.xlabel('Odor Class', fontsize=12)
plt.ylabel('AUC Score', fontsize=12)
plt.title('Performance by Class', fontsize=14, fontweight='bold')
plt.xticks(x_pos, LABELS, rotation=45)
plt.legend(fontsize=10)
plt.ylim(0, 1)
for i, (auc, pr_auc) in enumerate(zip(auc_scores, pr_auc_scores)):
    plt.text(i - width/2, auc + 0.01, f'{auc:.3f}', ha='center', va='bottom', fontsize=9)
    plt.text(i + width/2, pr_auc + 0.01, f'{pr_auc:.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
summary_path = os.path.join(PLOTS_DIR, 'performance_summary.png')
plt.savefig(summary_path, dpi=300, bbox_inches='tight')
plt.close()
print(f'Performance summary saved to: {summary_path}')

# Print summary metrics
print('\nSummary metrics:')
for label in LABELS:
    print(f"{label.capitalize()} - Precision: {results[f'{label}_precision']:.3f}, Recall: {results[f'{label}_recall']:.3f}, F1: {results[f'{label}_f1']:.3f}, ROC-AUC: {results[f'{label}_auc']:.3f}, PR-AUC: {results[f'{label}_pr_auc']:.3f}") 