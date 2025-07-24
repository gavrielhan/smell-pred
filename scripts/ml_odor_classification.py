#!/usr/bin/env python3
"""
Traditional ML Pipeline for Multi-Label Odor Classification

Uses LightGBM and XGBoost with molecular features extracted from SMILES
for predicting multiple odor labels: sweet, floral, minty, pungent.

Features:
- Morgan fingerprints and RDKit molecular descriptors
- GPU acceleration support for both LightGBM and XGBoost
- Multi-label classification with proper evaluation
- Comprehensive plotting and analysis
- Hyperparameter optimization
- Reproducible results with seed setting
"""

import pandas as pd
import numpy as np
import os
import argparse
import time
from datetime import datetime
import logging
import random
from typing import Dict, List, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# ML libraries
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score, roc_curve, 
    precision_recall_curve, hamming_loss, jaccard_score,
    multilabel_confusion_matrix, classification_report
)

# Chemistry libraries
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
    from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
    RDKIT_AVAILABLE = True
except ImportError:
    print("⚠️  RDKit not available. Please install: conda install -c conda-forge rdkit")
    RDKIT_AVAILABLE = False

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_header(title: str):
    """Print a formatted header for better readability"""
    print("\n" + "="*80)
    print(f"🧪 {title.upper()}")
    print("="*80)

def print_step(step: str, details: str = ""):
    """Print a formatted step indicator"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{timestamp}] 🔄 {step}")
    if details:
        print(f"   💡 {details}")

def print_success(message: str):
    """Print a success message"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] ✅ {message}")

def print_stats(title: str, stats: dict):
    """Print formatted statistics"""
    print(f"\n📊 {title}:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")

def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    print_step(f"Setting random seed for reproducibility", f"Seed: {seed}")
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # Set environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print_success(f"Random seed set to {seed} for all libraries")

def analyze_label_patterns(y_data: np.ndarray, label_names: List[str], title: str = "Dataset"):
    """Analyze multi-label patterns and correlations"""
    print_step(f"Analyzing multi-label patterns in {title}")
    
    # Basic label statistics
    label_counts = y_data.sum(axis=0)
    label_percentages = (label_counts / len(y_data)) * 100
    
    print(f"\n📊 {title} Label Distribution:")
    for i, label in enumerate(label_names):
        print(f"   {label.upper()}: {int(label_counts[i])} ({label_percentages[i]:.1f}%)")
    
    # Multi-label statistics
    num_labels_per_sample = y_data.sum(axis=1)
    unique_combinations = len(set([tuple(row) for row in y_data]))
    
    multilabel_stats = {
        'Total samples': len(y_data),
        'Samples with 0 labels': int((num_labels_per_sample == 0).sum()),
        'Samples with 1 label': int((num_labels_per_sample == 1).sum()),
        'Samples with 2+ labels': int((num_labels_per_sample >= 2).sum()),
        'Average labels per sample': float(num_labels_per_sample.mean()),
        'Max labels per sample': int(num_labels_per_sample.max()),
        'Unique label combinations': unique_combinations
    }
    print_stats(f"{title} Multi-Label Analysis", multilabel_stats)
    
    # Label correlations
    correlation_matrix = np.corrcoef(y_data.T)
    print(f"\n🔗 Label Correlations:")
    for i in range(len(label_names)):
        for j in range(i+1, len(label_names)):
            corr = correlation_matrix[i, j]
            print(f"   {label_names[i]} ↔ {label_names[j]}: {corr:.3f}")
    
    # Most common label combinations
    combinations = {}
    for row in y_data:
        combination = tuple(row)
        active_labels = [label_names[i] for i, val in enumerate(combination) if val == 1]
        if active_labels:  # Skip empty combinations
            combo_str = " + ".join(sorted(active_labels))
            combinations[combo_str] = combinations.get(combo_str, 0) + 1
    
    print(f"\n🏷️  Most Common Label Combinations:")
    sorted_combos = sorted(combinations.items(), key=lambda x: x[1], reverse=True)
    for combo, count in sorted_combos[:10]:  # Top 10
        percentage = (count / len(y_data)) * 100
        print(f"   {combo}: {count} ({percentage:.1f}%)")
    
    return {
        'label_counts': label_counts,
        'label_percentages': label_percentages,
        'correlation_matrix': correlation_matrix,
        'combinations': combinations,
        'multilabel_stats': multilabel_stats
    }

class MLOdorConfig:
    """Configuration for ML odor classification pipeline"""
    
    # Model configuration
    LABEL_NAMES = ['sweet', 'floral', 'minty', 'pungent']
    MODELS = ['lightgbm', 'xgboost']  # Available models
    
    # Feature extraction
    MORGAN_RADIUS = 2
    MORGAN_NBITS = 2048
    USE_FEATURES = True  # Use RDKit molecular descriptors
    USE_FINGERPRINTS = True  # Use Morgan fingerprints
    
    # Training configuration
    OUTPUT_DIR = "./ml_odor_results"
    RANDOM_SEED = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    
    # Model hyperparameters
    LIGHTGBM_PARAMS = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'device_type': 'cpu',  # Will be updated for GPU
        'n_estimators': 500
    }
    
    XGBOOST_PARAMS = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'tree_method': 'hist',  # Will be updated for GPU
        'n_estimators': 500
    }
    
    # Data files
    TRAIN_FILE = "data/goodscents_train.csv"
    TEST_FILE = "data/goodscents_test.csv"
    PREDICT_FILE = "data/bushdid_predict.csv"

def extract_molecular_features(smiles_list: List[str], config: MLOdorConfig) -> np.ndarray:
    """Extract molecular features from SMILES strings"""
    print_step("Extracting molecular features from SMILES")
    
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for feature extraction")
    
    features_list = []
    valid_indices = []
    
    # Use MorganGenerator if available (RDKit >=2023.03)
    try:
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
        morgan_generator = GetMorganGenerator(radius=config.MORGAN_RADIUS, fpSize=config.MORGAN_NBITS)
        use_morgan_generator = True
    except ImportError:
        use_morgan_generator = False
    
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"⚠️  Invalid SMILES: {smiles}")
            continue
            
        feature_vector = []
        
        # Morgan fingerprints
        if config.USE_FINGERPRINTS:
            if use_morgan_generator:
                fp = morgan_generator.GetFingerprint(mol)
                morgan_fp = list(fp.ToBitString())
                morgan_fp = [int(x) for x in morgan_fp]
            else:
                morgan_fp = GetMorganFingerprintAsBitVect(
                    mol, config.MORGAN_RADIUS, nBits=config.MORGAN_NBITS
                )
                morgan_fp = list(morgan_fp)
            feature_vector.extend(morgan_fp)
        
        # RDKit molecular descriptors
        if config.USE_FEATURES:
            try:
                # Basic molecular properties
                mw = Descriptors.MolWt(mol)
                logp = Crippen.MolLogP(mol)
                hbd = Lipinski.NumHDonors(mol)
                hba = Lipinski.NumHAcceptors(mol)
                tpsa = Descriptors.TPSA(mol)
                rotatable_bonds = Descriptors.NumRotatableBonds(mol)
                aromatic_rings = Descriptors.NumAromaticRings(mol)
                
                # Additional descriptors
                heavy_atoms = Descriptors.HeavyAtomCount(mol)
                formal_charge = Chem.rdmolops.GetFormalCharge(mol)
                
                # Lipinski's rule of five
                lipinski_violations = (
                    (mw > 500) + (logp > 5) + (hbd > 5) + (hba > 10)
                )
                
                descriptor_features = [
                    mw, logp, hbd, hba, tpsa, rotatable_bonds, 
                    aromatic_rings, heavy_atoms, formal_charge, 
                    lipinski_violations
                ]
                
                feature_vector.extend(descriptor_features)
                
            except Exception as e:
                print(f"⚠️  Error computing descriptors for {smiles}: {e}")
                continue
        
        features_list.append(feature_vector)
        valid_indices.append(i)
    
    features_array = np.array(features_list)
    
    # Feature statistics
    feature_stats = {
        'Total SMILES': len(smiles_list),
        'Valid molecules': len(features_list),
        'Invalid molecules': len(smiles_list) - len(features_list),
        'Feature dimensions': features_array.shape[1] if len(features_list) > 0 else 0,
        'Morgan fingerprint bits': config.MORGAN_NBITS if config.USE_FINGERPRINTS else 0,
        'Molecular descriptors': 10 if config.USE_FEATURES else 0
    }
    print_stats("Feature Extraction", feature_stats)
    
    return features_array, valid_indices

def setup_gpu_acceleration(model_name: str, params: dict) -> dict:
    """Configure GPU acceleration for ML models"""
    print_step(f"Configuring GPU acceleration for {model_name}")
    
    gpu_params = params.copy()
    
    if model_name == 'lightgbm':
        try:
            # Try to use GPU for LightGBM
            gpu_params['device_type'] = 'gpu'
            gpu_params['gpu_platform_id'] = 0
            gpu_params['gpu_device_id'] = 0
            print_success("LightGBM GPU acceleration configured")
        except Exception as e:
            print(f"⚠️  LightGBM GPU not available, using CPU: {e}")
            gpu_params['device_type'] = 'cpu'
    
    elif model_name == 'xgboost':
        try:
            # Try to use GPU for XGBoost
            gpu_params['tree_method'] = 'gpu_hist'
            gpu_params['gpu_id'] = 0
            print_success("XGBoost GPU acceleration configured")
        except Exception as e:
            print(f"⚠️  XGBoost GPU not available, using CPU: {e}")
            gpu_params['tree_method'] = 'hist'
    
    return gpu_params

def create_ml_model(model_name: str, config: MLOdorConfig, use_gpu: bool = False):
    """Create and configure ML model"""
    print_step(f"Creating {model_name} model")
    
    if model_name == 'lightgbm':
        params = config.LIGHTGBM_PARAMS.copy()
        if use_gpu:
            params = setup_gpu_acceleration('lightgbm', params)
        
        # Create multi-output wrapper
        base_model = lgb.LGBMClassifier(**params)
        model = MultiOutputClassifier(base_model, n_jobs=-1)
        
    elif model_name == 'xgboost':
        params = config.XGBOOST_PARAMS.copy()
        if use_gpu:
            params = setup_gpu_acceleration('xgboost', params)
        
        # Create multi-output wrapper
        base_model = xgb.XGBClassifier(**params)
        model = MultiOutputClassifier(base_model, n_jobs=-1)
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

def train_and_evaluate_model(model, X_train: np.ndarray, X_test: np.ndarray,
                           y_train: np.ndarray, y_test: np.ndarray,
                           model_name: str, config: MLOdorConfig) -> Dict:
    """Train model and compute detailed evaluation metrics"""
    print_step(f"Training {model_name} model")
    
    # Train the model
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print_success(f"{model_name} training completed in {training_time:.2f} seconds")
    
    # Make predictions
    print_step("Making predictions and computing metrics")
    y_pred_proba = model.predict_proba(X_test)
    
    # Handle multi-output predictions
    if len(y_pred_proba) == len(config.LABEL_NAMES):
        # Extract probabilities for positive class
        y_pred_proba_positive = np.column_stack([pred_proba[:, 1] for pred_proba in y_pred_proba])
    else:
        y_pred_proba_positive = y_pred_proba
    
    y_pred_binary = (y_pred_proba_positive > 0.5).astype(int)
    
    # Compute metrics
    results = {}
    
    # Overall multi-label metrics
    results['model_name'] = model_name
    results['training_time'] = training_time
    results['exact_match'] = accuracy_score(y_test, y_pred_binary)  # Subset accuracy
    results['hamming_loss'] = hamming_loss(y_test, y_pred_binary)
    results['jaccard_score'] = jaccard_score(y_test, y_pred_binary, average='samples')
    
    # Additional multi-label specific metrics
    results['jaccard_macro'] = jaccard_score(y_test, y_pred_binary, average='macro')
    results['jaccard_micro'] = jaccard_score(y_test, y_pred_binary, average='micro')
    
    # Label-wise accuracy (considering only samples where the label appears)
    label_wise_accuracies = []
    for i in range(len(config.LABEL_NAMES)):
        # Only consider samples where this label appears in ground truth
        label_mask = y_test[:, i] == 1
        if label_mask.sum() > 0:
            label_accuracy = accuracy_score(y_test[label_mask, i], y_pred_binary[label_mask, i])
            label_wise_accuracies.append(label_accuracy)
        else:
            label_wise_accuracies.append(0.0)
    
    results['mean_label_accuracy'] = np.mean(label_wise_accuracies)
    
    # Micro and macro averaged metrics
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_test, y_pred_binary, average='micro', zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred_binary, average='macro', zero_division=0
    )
    
    results.update({
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro
    })
    
    # Per-class metrics
    for i, label in enumerate(config.LABEL_NAMES):
        y_true_label = y_test[:, i]
        y_pred_label = y_pred_binary[:, i]
        y_prob_label = y_pred_proba_positive[:, i]
        
        if len(np.unique(y_true_label)) > 1:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true_label, y_pred_label, average='binary', zero_division=0
            )
            auc = roc_auc_score(y_true_label, y_prob_label)
            pr_auc = average_precision_score(y_true_label, y_prob_label)
        else:
            precision = recall = f1 = auc = pr_auc = 0.0
        
        results.update({
            f'{label}_precision': precision,
            f'{label}_recall': recall,
            f'{label}_f1': f1,
            f'{label}_auc': auc,
            f'{label}_pr_auc': pr_auc
        })
    
    # Store predictions for plotting
    results['y_true'] = y_test
    results['y_pred_proba'] = y_pred_proba_positive
    results['y_pred_binary'] = y_pred_binary
    
    return results

def create_ml_plots(results_dict: Dict, config: MLOdorConfig, output_dir: str):
    """Create comprehensive plots for ML model results"""
    print_step("Creating performance plots for ML models")
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create output directory for plots
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Model comparison - F1 scores
    print("📊 Creating model comparison plots...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # F1 scores comparison
    ax = axes[0, 0]
    models = list(results_dict.keys())
    f1_micro = [results_dict[model]['f1_micro'] for model in models]
    f1_macro = [results_dict[model]['f1_macro'] for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax.bar(x - width/2, f1_micro, width, label='Micro F1', alpha=0.8)
    ax.bar(x + width/2, f1_macro, width, label='Macro F1', alpha=0.8)
    ax.set_xlabel('Model')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Per-class F1 scores
    ax = axes[0, 1]
    for i, model in enumerate(models):
        f1_scores = [results_dict[model][f'{label}_f1'] for label in config.LABEL_NAMES]
        ax.plot(config.LABEL_NAMES, f1_scores, marker='o', label=model, linewidth=2)
    
    ax.set_xlabel('Odor Class')
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Class F1 Scores')
    ax.legend()
    ax.set_ylim(0, 1)
    plt.setp(ax.get_xticklabels(), rotation=45)
    
    # AUC scores comparison
    ax = axes[1, 0]
    for i, model in enumerate(models):
        auc_scores = [results_dict[model][f'{label}_auc'] for label in config.LABEL_NAMES]
        ax.plot(config.LABEL_NAMES, auc_scores, marker='s', label=model, linewidth=2)
    
    ax.set_xlabel('Odor Class')
    ax.set_ylabel('ROC-AUC')
    ax.set_title('Per-Class ROC-AUC Scores')
    ax.legend()
    ax.set_ylim(0, 1)
    plt.setp(ax.get_xticklabels(), rotation=45)
    
    # Training time comparison
    ax = axes[1, 1]
    training_times = [results_dict[model]['training_time'] for model in models]
    bars = ax.bar(models, training_times, alpha=0.7)
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Training Time Comparison')
    
    # Add value labels on bars
    for bar, time_val in zip(bars, training_times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{time_val:.1f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    comparison_path = os.path.join(plots_dir, "model_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   💾 Model comparison saved to: {comparison_path}")
    
    # 2. ROC and PR curves for all models
    for model in models:
        results = results_dict[model]
        print(f"📊 Creating ROC and PR curves for model: {model}")
        plt.figure(figsize=(12, 8))
        colors = sns.color_palette("husl", len(config.LABEL_NAMES))
        for i, (label, color) in enumerate(zip(config.LABEL_NAMES, colors)):
            y_true = results['y_true'][:, i]
            y_prob = results['y_pred_proba'][:, i]
            if len(np.unique(y_true)) > 1:
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                auc_score = roc_auc_score(y_true, y_prob)
                plt.plot(fpr, tpr, color=color, linewidth=2,
                        label=f'{label.upper()} (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curves - {model.upper()}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        roc_path = os.path.join(plots_dir, f"roc_curves_{model}.png")
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   💾 ROC curves saved to: {roc_path}")
        # PR-AUC curves
        plt.figure(figsize=(12, 8))
        for i, (label, color) in enumerate(zip(config.LABEL_NAMES, colors)):
            y_true = results['y_true'][:, i]
            y_prob = results['y_pred_proba'][:, i]
            if len(np.unique(y_true)) > 1:
                precision, recall, _ = precision_recall_curve(y_true, y_prob)
                pr_auc = average_precision_score(y_true, y_prob)
                plt.plot(recall, precision, color=color, linewidth=2,
                        label=f'{label.upper()} (PR-AUC = {pr_auc:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curves - {model.upper()}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        pr_path = os.path.join(plots_dir, f"precision_recall_curves_{model}.png")
        plt.savefig(pr_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   💾 PR curves saved to: {pr_path}")
    
    # 3. Multi-label pattern analysis (unchanged)
    print("📊 Creating multi-label pattern analysis...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    best_model = max(results_dict.keys(), key=lambda k: results_dict[k]['f1_macro'])
    best_results = results_dict[best_model]
    y_true = best_results['y_true']
    y_pred = best_results['y_pred_binary']
    
    # Label co-occurrence heatmap (true labels)
    ax = axes[0, 0]
    cooccurrence = np.dot(y_true.T, y_true)
    sns.heatmap(cooccurrence, annot=True, fmt='d', cmap='Blues',
                xticklabels=config.LABEL_NAMES, yticklabels=config.LABEL_NAMES, ax=ax)
    ax.set_title('True Label Co-occurrence Matrix')
    
    # Prediction vs truth confusion for multi-label
    ax = axes[0, 1]
    # Hamming distance distribution
    hamming_distances = []
    for i in range(len(y_true)):
        hamming_dist = np.sum(y_true[i] != y_pred[i])
        hamming_distances.append(hamming_dist)
    
    unique_distances, counts = np.unique(hamming_distances, return_counts=True)
    ax.bar(unique_distances, counts, alpha=0.7)
    ax.set_xlabel('Hamming Distance (# wrong labels)')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Prediction Error Distribution')
    ax.set_xticks(unique_distances)
    
    # Number of labels per sample
    ax = axes[1, 0]
    true_label_counts = y_true.sum(axis=1)
    pred_label_counts = y_pred.sum(axis=1)
    
    bins = np.arange(0, max(true_label_counts.max(), pred_label_counts.max()) + 2) - 0.5
    ax.hist(true_label_counts, bins=bins, alpha=0.7, label='True', density=True)
    ax.hist(pred_label_counts, bins=bins, alpha=0.7, label='Predicted', density=True)
    ax.set_xlabel('Number of Labels per Sample')
    ax.set_ylabel('Density')
    ax.set_title('Label Count Distribution')
    ax.legend()
    ax.set_xticks(range(int(bins.max())))
    
    # Subset accuracy by number of true labels
    ax = axes[1, 1]
    subset_accuracies = []
    label_count_groups = []
    
    for num_labels in range(0, true_label_counts.max() + 1):
        mask = true_label_counts == num_labels
        if mask.sum() > 0:
            subset_acc = accuracy_score(y_true[mask], y_pred[mask])
            subset_accuracies.append(subset_acc)
            label_count_groups.append(num_labels)
    
    ax.bar(label_count_groups, subset_accuracies, alpha=0.7)
    ax.set_xlabel('Number of True Labels')
    ax.set_ylabel('Subset Accuracy')
    ax.set_title('Accuracy by Label Complexity')
    ax.set_ylim(0, 1)
    
    # Add sample counts as annotations
    for i, (num_labels, acc) in enumerate(zip(label_count_groups, subset_accuracies)):
        sample_count = (true_label_counts == num_labels).sum()
        ax.text(i, acc + 0.02, f'n={sample_count}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    multilabel_path = os.path.join(plots_dir, "multilabel_analysis.png")
    plt.savefig(multilabel_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   💾 Multi-label analysis saved to: {multilabel_path}")
    
    print_success(f"All plots saved to: {plots_dir}")

def run_ml_pipeline(config: MLOdorConfig, models_to_run: List[str], use_gpu: bool = False) -> Dict:
    """Run the complete ML pipeline"""
    print_header("ML PIPELINE FOR ODOR CLASSIFICATION")
    
    # Set random seed
    set_seed(config.RANDOM_SEED)
    
    # Print configuration
    config_info = {
        'Target labels': ', '.join(config.LABEL_NAMES),
        'Models to run': ', '.join(models_to_run),
        'Morgan fingerprint bits': config.MORGAN_NBITS,
        'Use molecular descriptors': config.USE_FEATURES,
        'Use fingerprints': config.USE_FINGERPRINTS,
        'GPU acceleration': use_gpu,
        'Random seed': config.RANDOM_SEED,
        'Output directory': config.OUTPUT_DIR
    }
    print_stats("Pipeline Configuration", config_info)
    
    # Create output directory
    print_step("Creating output directories")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    print_success(f"Directory created: {config.OUTPUT_DIR}")
    
    # Load datasets
    print_step("Loading datasets")
    train_df = pd.read_csv(config.TRAIN_FILE)
    test_df = pd.read_csv(config.TEST_FILE)
    
    print_stats("Dataset Information", {
        'Training samples': len(train_df),
        'Test samples': len(test_df),
        'Total samples': len(train_df) + len(test_df)
    })
    
    # Extract features
    print_step("Feature extraction pipeline")
    
    # Combine train and test for feature extraction
    all_smiles = list(train_df['IsomericSMILES']) + list(test_df['IsomericSMILES'])
    all_features, valid_indices = extract_molecular_features(all_smiles, config)
    
    if len(all_features) == 0:
        raise ValueError("No valid molecular features extracted!")
    
    # Split features back to train and test
    n_train = len(train_df)
    train_indices = [i for i in valid_indices if i < n_train]
    test_indices = [i - n_train for i in valid_indices if i >= n_train]
    
    X_train = all_features[:len(train_indices)]
    X_test = all_features[len(train_indices):]
    
    # Extract labels
    y_train = train_df.iloc[train_indices][config.LABEL_NAMES].values
    y_test = test_df.iloc[test_indices][config.LABEL_NAMES].values
    
    print_stats("Feature Matrix", {
        'Training features shape': X_train.shape,
        'Test features shape': X_test.shape,
        'Training labels shape': y_train.shape,
        'Test labels shape': y_test.shape
    })
    
    # Analyze multi-label patterns
    train_analysis = analyze_label_patterns(y_train, config.LABEL_NAMES, "Training Set")
    test_analysis = analyze_label_patterns(y_test, config.LABEL_NAMES, "Test Set")
    
    # Train and evaluate models
    results_dict = {}
    
    for model_name in models_to_run:
        print_header(f"TRAINING {model_name.upper()}")
        
        # Create and train model
        model = create_ml_model(model_name, config, use_gpu)
        results = train_and_evaluate_model(
            model, X_train, X_test, y_train, y_test, model_name, config
        )
        
        results_dict[model_name] = results
        
        # Print results
        print_stats(f"{model_name} Results", {
            'F1 Macro': results['f1_macro'],
            'F1 Micro': results['f1_micro'],
            'Exact Match': results['exact_match'],
            'Hamming Loss': results['hamming_loss'],
            'Training Time': f"{results['training_time']:.2f}s"
        })
    
    # Create plots
    create_ml_plots(results_dict, config, config.OUTPUT_DIR)
    
    # Save results
    results_df = pd.DataFrame([
        {
            'model': model_name,
            'f1_macro': results['f1_macro'],
            'f1_micro': results['f1_micro'],
            'exact_match': results['exact_match'],
            'hamming_loss': results['hamming_loss'],
            'training_time': results['training_time'],
            **{f'{label}_f1': results[f'{label}_f1'] for label in config.LABEL_NAMES},
            **{f'{label}_precision': results[f'{label}_precision'] for label in config.LABEL_NAMES},
            **{f'{label}_recall': results[f'{label}_recall'] for label in config.LABEL_NAMES},
            **{f'{label}_auc': results[f'{label}_auc'] for label in config.LABEL_NAMES},
            **{f'{label}_pr_auc': results[f'{label}_pr_auc'] for label in config.LABEL_NAMES},
        }
        for model_name, results in results_dict.items()
    ])
    
    results_path = os.path.join(config.OUTPUT_DIR, "model_results.csv")
    results_df.to_csv(results_path, index=False)
    print_success(f"Results saved to: {results_path}")
    
    return results_dict

def main():
    parser = argparse.ArgumentParser(description="ML Pipeline for odor classification")
    parser.add_argument("--models", nargs='+', choices=['lightgbm', 'xgboost'], 
                       default=['lightgbm', 'xgboost'], help="Models to train")
    parser.add_argument("--gpu", action='store_true', help="Use GPU acceleration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./ml_odor_results", 
                       help="Output directory")
    parser.add_argument("--morgan_bits", type=int, default=2048, 
                       help="Morgan fingerprint bits")
    parser.add_argument("--no_features", action='store_true', 
                       help="Skip molecular descriptors")
    parser.add_argument("--no_fingerprints", action='store_true', 
                       help="Skip Morgan fingerprints")
    
    args = parser.parse_args()
    
    if args.no_features and args.no_fingerprints:
        raise ValueError("Cannot disable both features and fingerprints!")
    
    # Print configuration
    print_header("COMMAND LINE ARGUMENTS")
    arg_stats = {
        'Models': ', '.join(args.models),
        'GPU acceleration': args.gpu,
        'Random seed': args.seed,
        'Morgan fingerprint bits': args.morgan_bits,
        'Use molecular descriptors': not args.no_features,
        'Use fingerprints': not args.no_fingerprints,
        'Output directory': args.output_dir
    }
    print_stats("Arguments", arg_stats)
    
    # Update config
    config = MLOdorConfig()
    config.RANDOM_SEED = args.seed
    config.OUTPUT_DIR = args.output_dir
    config.MORGAN_NBITS = args.morgan_bits
    config.USE_FEATURES = not args.no_features
    config.USE_FINGERPRINTS = not args.no_fingerprints
    
    # Check RDKit availability
    if not RDKIT_AVAILABLE:
        print("❌ RDKit is required for this pipeline")
        print("Install with: conda install -c conda-forge rdkit")
        return
    
    # Run pipeline
    print(f"\n🎯 Starting ML pipeline with configuration:")
    print(f"   📁 Output: {config.OUTPUT_DIR}")
    print(f"   🤖 Models: {', '.join(args.models)}")
    print(f"   ⚡ GPU: {args.gpu}")
    print(f"   🎲 Random seed: {config.RANDOM_SEED}")
    print(f"   🧬 Feature extraction: Morgan FP + RDKit descriptors")
    
    start_time = time.time()
    results = run_ml_pipeline(config, args.models, args.gpu)
    total_time = time.time() - start_time
    
    # Print final summary
    print_header("🎉 SUCCESS!")
    print(f"🎯 ML pipeline completed successfully!")
    print(f"⏱️  Total execution time: {total_time/60:.2f} minutes")
    print(f"📁 Results saved to: {config.OUTPUT_DIR}")
    
    # Show best model
    best_model = max(results.keys(), key=lambda k: results[k]['f1_macro'])
    print(f"🏆 Best model: {best_model} (F1 Macro: {results[best_model]['f1_macro']:.4f})")
    print("="*80)

if __name__ == "__main__":
    main() 