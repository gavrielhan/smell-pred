#!/usr/bin/env python3
"""
ChemBERTa LoRA Fine-tuning for Multi-Label Odor Classification

Uses LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning of 
ChemBERTa for predicting multiple odor labels: sweet, floral, minty, pungent.

Inspired by FART (Flavor Analysis and Recognition Transformer), which uses 
the same approach (ChemBERTa + fine-tuning) but for taste prediction.
Our adaptation focuses on odor classification instead.

LoRA Benefits:
- 90% less memory usage
- Faster training with only ~1-5% of parameters
- Better generalization and less overfitting
- Can run efficiently on consumer hardware
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, TrainerCallback
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from typing import Dict, List, Tuple
import logging
import time
from datetime import datetime
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
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
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may be slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # For MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    # Set environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print_success(f"Random seed set to {seed} for all libraries")

class OdorChemBERTaConfig:
    """Configuration for ChemBERTa LoRA fine-tuning for odor classification"""
    
    # Model configuration
    MODEL_CHECKPOINT = "seyonec/SMILES_tokenized_PubChem_shard00_160k"  # ChemBERTa base
    NUM_LABELS = 4
    LABEL_NAMES = ['sweet', 'floral', 'minty', 'pungent']
    MAX_LENGTH = 256  # Optimized for memory efficiency
    
    # LoRA configuration - Parameter-efficient fine-tuning
    LORA_R = 16  # Rank of adaptation, higher = more parameters but better performance
    LORA_ALPHA = 32  # LoRA scaling parameter (usually 2*r)
    LORA_DROPOUT = 0.1  # Dropout for LoRA layers
    LORA_TARGET_MODULES = ["query", "value"]  # Apply LoRA to attention layers
    LORA_BIAS = "none"  # Don't train bias parameters
    
    # Training configuration - Optimized for LoRA + MacBook Air M2 8GB
    OUTPUT_DIR = "./chemberta_lora_results"
    LOGGING_DIR = "./chemberta_lora_logs"
    NUM_EPOCHS = 50  # Extended training for better convergence
    BATCH_SIZE = 16  # Can use larger batch size with LoRA
    LEARNING_RATE = 5e-4  # Reduced LR to prevent overfitting over 20 epochs
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 100  # More warmup for longer training
    
    # Data files
    TRAIN_FILE = "data/goodscents_train.csv"
    TEST_FILE = "data/goodscents_test.csv"
    PREDICT_FILE = "data/bushdid_predict.csv"
    
    # Reproducibility
    RANDOM_SEED = 42  # For reproducible results

class TrainingHistoryCallback(TrainerCallback):
    """Custom callback to track training history at every log step"""
    def __init__(self):
        self.training_history = {
            'step': [],
            'epoch': [],
            'train_loss': [],
            'eval_loss': [],
            'eval_f1_macro': [],
            'eval_f1_micro': [],
            'learning_rate': []
        }
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Only log if step is present (i.e., not just at init)
        if logs is None or 'step' not in logs:
            return
        step = logs.get('step')
        epoch = logs.get('epoch', None)
        train_loss = logs.get('loss', None)
        eval_loss = logs.get('eval_loss', None)
        eval_f1_macro = logs.get('eval_f1_macro', None)
        eval_f1_micro = logs.get('eval_f1_micro', None)
        lr = logs.get('learning_rate', None)
        self.training_history['step'].append(step)
        self.training_history['epoch'].append(epoch)
        self.training_history['train_loss'].append(train_loss)
        self.training_history['eval_loss'].append(eval_loss)
        self.training_history['eval_f1_macro'].append(eval_f1_macro)
        self.training_history['eval_f1_micro'].append(eval_f1_micro)
        self.training_history['learning_rate'].append(lr)
        print(f"[DEBUG] step: {step}, epoch: {epoch}, train_loss: {train_loss}, eval_loss: {eval_loss}, eval_f1_macro: {eval_f1_macro}, eval_f1_micro: {eval_f1_micro}, lr: {lr}")
    def get_history(self):
        return self.training_history

class MultiLabelOdorTrainer(Trainer):
    """Custom trainer for multi-label odor classification using BCE loss"""
    
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_count = 0
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute BCE loss for multi-label classification with optional class weights"""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        # Use BCE with logits loss for multi-label classification
        if self.class_weights is not None:
            # Apply class weights to handle imbalanced data
            weights = torch.tensor(self.class_weights, device=logits.device, dtype=logits.dtype)
            # Expand weights to match batch size
            weights = weights.unsqueeze(0).expand(labels.size(0), -1)
            loss_fct = BCEWithLogitsLoss(weight=weights, reduction='mean')
        else:
            loss_fct = BCEWithLogitsLoss()
        
        loss = loss_fct(logits, labels.float())
        
        return (loss, outputs) if return_outputs else loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluate to add progress tracking"""
        self.eval_count += 1
        print(f"\n🔍 Running evaluation #{self.eval_count}...")
        
        start_time = time.time()
        results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        eval_time = time.time() - start_time
        
        print(f"⏱️  Evaluation completed in {eval_time:.2f} seconds")
        
        # Print key metrics
        key_metrics = {k: v for k, v in results.items() if any(metric in k for metric in ['f1_macro', 'exact_match', 'loss'])}
        print_stats("Evaluation Metrics", key_metrics)
        
        return results
    
    def log(self, logs: Dict[str, float], start_time=None) -> None:
        """Override log to add training progress prints"""
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)
        # Print training progress
        step = logs.get('step', None)
        if step is None:
            # Fallback: use self.state.global_step if available
            step = getattr(self.state, 'global_step', None)
        if 'loss' in logs and step is not None and step % 25 == 0:  # Every 25 steps
            loss = logs.get('loss', 0)
            lr = logs.get('learning_rate', 0)
            print(f"📈 Step {step}: Loss = {loss:.4f}, LR = {lr:.2e}")

def load_datasets(config: OdorChemBERTaConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, test, and prediction datasets"""
    print_step("Loading datasets", f"Train: {config.TRAIN_FILE}, Test: {config.TEST_FILE}")
    
    start_time = time.time()
    
    try:
        train_df = pd.read_csv(config.TRAIN_FILE)
        test_df = pd.read_csv(config.TEST_FILE)
        predict_df = pd.read_csv(config.PREDICT_FILE)
    except FileNotFoundError as e:
        print(f"❌ Error loading datasets: {e}")
        raise
    
    load_time = time.time() - start_time
    print_success(f"Datasets loaded in {load_time:.2f} seconds")
    
    # Dataset statistics
    dataset_stats = {
        'Train samples': len(train_df),
        'Test samples': len(test_df),
        'Prediction samples': len(predict_df),
        'Total molecules': len(train_df) + len(test_df) + len(predict_df)
    }
    print_stats("Dataset Statistics", dataset_stats)
    
    # Check for required columns
    print("\n🔍 Checking dataset columns...")
    required_cols = ['IsomericSMILES'] + config.LABEL_NAMES
    for df_name, df in [('Train', train_df), ('Test', test_df)]:
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ {df_name} dataset missing columns: {missing_cols}")
        else:
            print(f"✅ {df_name} dataset has all required columns")
    
    # Show label distribution in training set
    print_step("Analyzing label distribution")
    label_stats = {}
    for label in config.LABEL_NAMES:
        if label in train_df.columns:
            count = train_df[label].sum()
            pct = (count / len(train_df)) * 100
            label_stats[f'{label.upper()}'] = f"{count} ({pct:.1f}%)"
            
    print_stats("Training Set Label Distribution", label_stats)
    
    # Check for missing SMILES
    train_missing = train_df['IsomericSMILES'].isna().sum()
    test_missing = test_df['IsomericSMILES'].isna().sum()
    if train_missing > 0 or test_missing > 0:
        print(f"⚠️  Missing SMILES - Train: {train_missing}, Test: {test_missing}")
    else:
        print("✅ No missing SMILES found")
    
    return train_df, test_df, predict_df

def tokenize_smiles(examples: Dict, tokenizer, max_length: int = 512) -> Dict:
    """Tokenize SMILES strings for transformer input"""
    return tokenizer(
        examples["IsomericSMILES"], 
        padding="max_length", 
        truncation=True, 
        max_length=max_length
    )

def prepare_datasets(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                    tokenizer, config: OdorChemBERTaConfig) -> Tuple[Dataset, Dataset]:
    """Convert pandas DataFrames to HuggingFace Datasets with tokenization"""
    print_step("Preparing datasets for training", "Converting to HuggingFace Dataset format")
    
    start_time = time.time()
    
    # Convert to HuggingFace Dataset format
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    print_success("Converted DataFrames to HuggingFace Datasets")
    
    # Tokenize SMILES
    print_step("Tokenizing SMILES", f"Max length: {config.MAX_LENGTH}")
    tokenize_start = time.time()
    
    train_dataset = train_dataset.map(
        lambda x: tokenize_smiles(x, tokenizer, config.MAX_LENGTH), 
        batched=True
    )
    test_dataset = test_dataset.map(
        lambda x: tokenize_smiles(x, tokenizer, config.MAX_LENGTH), 
        batched=True
    )
    
    tokenize_time = time.time() - tokenize_start
    print_success(f"Tokenization completed in {tokenize_time:.2f} seconds")
    
    # Show tokenization statistics
    sample_tokens = train_dataset[0]['input_ids']
    token_stats = {
        'Max sequence length': config.MAX_LENGTH,
        'Sample token count': len([t for t in sample_tokens if t != tokenizer.pad_token_id]),
        'Vocabulary size': tokenizer.vocab_size,
        'Padding token ID': tokenizer.pad_token_id
    }
    print_stats("Tokenization Statistics", token_stats)
    
    # Prepare multi-label targets
    print_step("Formatting labels", "Converting to multi-label tensor format")
    
    def format_labels(examples):
        """Convert multi-label columns to tensor format"""
        labels = []
        for i in range(len(examples[config.LABEL_NAMES[0]])):
            label_vector = [float(examples[label][i]) for label in config.LABEL_NAMES]  # Convert to float
            labels.append(label_vector)
        return {"labels": labels}
    
    train_dataset = train_dataset.map(format_labels, batched=True)
    test_dataset = test_dataset.map(format_labels, batched=True)
    
    # Set format for PyTorch
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    prep_time = time.time() - start_time
    print_success(f"Dataset preparation completed in {prep_time:.2f} seconds")
    
    # Final dataset statistics
    final_stats = {
        'Train samples': len(train_dataset),
        'Test samples': len(test_dataset),
        'Features per sample': len(train_dataset.column_names),
        'Label dimensions': len(config.LABEL_NAMES)
    }
    print_stats("Final Dataset Statistics", final_stats)
    
    # Show sample data structure
    print("\n🔍 Sample data structure:")
    sample = train_dataset[0]
    for key, value in sample.items():
        if hasattr(value, 'shape'):
            print(f"   {key}: {value.shape} {value.dtype}")
        else:
            print(f"   {key}: {type(value)}")
    
    return train_dataset, test_dataset

def compute_multi_label_metrics(eval_pred, label_names: List[str]) -> Dict:
    """Compute comprehensive metrics for multi-label classification"""
    print("\n🧮 Computing evaluation metrics...")
    
    logits, labels = eval_pred
    
    # Convert logits to probabilities using sigmoid
    predictions_proba = torch.sigmoid(torch.tensor(logits)).numpy()
    
    # Convert to binary predictions (threshold = 0.5)
    predictions_binary = (predictions_proba > 0.5).astype(int)
    
    results = {}
    
    # Per-label metrics
    print("📊 Computing per-label metrics...")
    for i, label_name in enumerate(label_names):
        y_true = labels[:, i]
        y_pred = predictions_binary[:, i]
        y_proba = predictions_proba[:, i]
        
        # Show label statistics
        true_count = y_true.sum()
        pred_count = y_pred.sum()
        print(f"   {label_name}: {true_count} true labels, {pred_count} predictions")
        
        # Skip if no positive samples
        if y_true.sum() > 0:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
            try:
                auc = roc_auc_score(y_true, y_proba)
                pr_auc = average_precision_score(y_true, y_proba)
            except ValueError:
                auc = 0.0
                pr_auc = 0.0
            
            results[f'{label_name}_precision'] = precision
            results[f'{label_name}_recall'] = recall
            results[f'{label_name}_f1'] = f1
            results[f'{label_name}_auc'] = auc
            results[f'{label_name}_pr_auc'] = pr_auc
    
    # Overall metrics
    print("🎯 Computing overall metrics...")
    results['exact_match'] = (predictions_binary == labels).all(axis=1).mean()  # Subset accuracy
    results['hamming_loss'] = (predictions_binary != labels).mean()
    
    # Micro-averaged F1 (treats each label prediction independently)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        labels.flatten(), predictions_binary.flatten(), average='micro', zero_division=0
    )
    results['f1_micro'] = f1_micro
    results['precision_micro'] = precision_micro
    results['recall_micro'] = recall_micro
    
    # Macro-averaged F1 (average across labels)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions_binary, average='macro', zero_division=0
    )
    results['f1_macro'] = f1_macro
    results['precision_macro'] = precision_macro
    results['recall_macro'] = recall_macro
    
    print_success("Metrics computation completed")
    
    return results

def create_performance_plots(y_true: np.ndarray, y_proba: np.ndarray, 
                           label_names: List[str], output_dir: str, 
                           training_history: Dict = None) -> None:
    """Create comprehensive performance plots for the model"""
    print_step("Creating performance plots")
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create output directory for plots
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. ROC Curves for each class
    print("📊 Creating ROC curves...")
    plt.figure(figsize=(12, 8))
    
    colors = sns.color_palette("husl", len(label_names))
    
    for i, (label, color) in enumerate(zip(label_names, colors)):
        if len(np.unique(y_true[:, i])) > 1:  # Only plot if both classes present
            fpr, tpr, _ = roc_curve(y_true[:, i], y_proba[:, i])
            auc_score = roc_auc_score(y_true[:, i], y_proba[:, i])
            plt.plot(fpr, tpr, color=color, linewidth=2,
                    label=f'{label.upper()} (AUC = {auc_score:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves by Odor Class', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    roc_path = os.path.join(plots_dir, "roc_curves.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   💾 ROC curves saved to: {roc_path}")
    
    # 2. Precision-Recall Curves for each class
    print("📊 Creating PR curves...")
    plt.figure(figsize=(12, 8))
    
    for i, (label, color) in enumerate(zip(label_names, colors)):
        if len(np.unique(y_true[:, i])) > 1:  # Only plot if both classes present
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_proba[:, i])
            pr_auc = average_precision_score(y_true[:, i], y_proba[:, i])
            plt.plot(recall, precision, color=color, linewidth=2,
                    label=f'{label.upper()} (PR-AUC = {pr_auc:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves by Odor Class', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    pr_path = os.path.join(plots_dir, "precision_recall_curves.png")
    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   💾 PR curves saved to: {pr_path}")
    
    # 3. Training history plots (if available)
    if training_history:
        print("📊 Creating training history plots...")
        
        # Training and validation loss
        if 'train_loss' in training_history and 'eval_loss' in training_history:
            plt.figure(figsize=(15, 10))
            
            # Loss plot
            plt.subplot(2, 2, 1)
            if len(training_history['train_loss']) > 0:
                epochs = range(1, len(training_history['train_loss']) + 1)
                plt.plot(epochs, training_history['train_loss'], 'b-', linewidth=2, label='Training Loss', marker='o')
            if len(training_history['eval_loss']) > 0:
                eval_epochs = range(1, len(training_history['eval_loss']) + 1)
                plt.plot(eval_epochs, training_history['eval_loss'], 'r-', linewidth=2, label='Validation Loss', marker='s')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title('Training & Validation Loss', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            
            # F1 scores over time
            plt.subplot(2, 2, 2)
            if 'eval_f1_macro' in training_history and len(training_history['eval_f1_macro']) > 0:
                eval_epochs = range(1, len(training_history['eval_f1_macro']) + 1)
                plt.plot(eval_epochs, training_history['eval_f1_macro'], 'g-', linewidth=2, label='Macro F1', marker='o')
            if 'eval_f1_micro' in training_history and len(training_history['eval_f1_micro']) > 0:
                eval_epochs = range(1, len(training_history['eval_f1_micro']) + 1)
                plt.plot(eval_epochs, training_history['eval_f1_micro'], 'orange', linewidth=2, label='Micro F1', marker='s')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('F1 Score', fontsize=12)
            plt.title('F1 Scores Over Training', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            
            # Learning rate schedule
            plt.subplot(2, 2, 3)
            if 'learning_rate' in training_history and len(training_history['learning_rate']) > 0:
                lr_epochs = range(1, len(training_history['learning_rate']) + 1)
                plt.plot(lr_epochs, training_history['learning_rate'], 'purple', linewidth=2, label='Learning Rate', marker='o')
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel('Learning Rate', fontsize=12)
                plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3)
                plt.yscale('log')  # Log scale for learning rate
            
            # Loss difference (overfitting indicator)
            plt.subplot(2, 2, 4)
            if (len(training_history['train_loss']) > 0 and len(training_history['eval_loss']) > 0 and
                len(training_history['train_loss']) == len(training_history['eval_loss'])):
                epochs = range(1, len(training_history['train_loss']) + 1)
                loss_diff = np.array(training_history['eval_loss']) - np.array(training_history['train_loss'])
                plt.plot(epochs, loss_diff, 'red', linewidth=2, label='Validation - Training Loss', marker='o')
                plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel('Loss Difference', fontsize=12)
                plt.title('Overfitting Indicator', fontsize=14, fontweight='bold')
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3)
                
                # Add annotation about overfitting
                if len(loss_diff) > 0 and loss_diff[-1] > 0.1:
                    plt.text(0.05, 0.95, 'Possible Overfitting', transform=plt.gca().transAxes,
                            bbox=dict(boxstyle='round', facecolor='red', alpha=0.3),
                            fontsize=10, verticalalignment='top')
            
            plt.tight_layout()
            history_path = os.path.join(plots_dir, "training_history.png")
            plt.savefig(history_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   💾 Training history saved to: {history_path}")
    
    # 4. Class distribution and performance summary
    print("📊 Creating performance summary...")
    plt.figure(figsize=(14, 6))
    
    # Class distribution
    plt.subplot(1, 2, 1)
    class_counts = y_true.sum(axis=0)
    bars = plt.bar(label_names, class_counts, color=colors)
    plt.xlabel('Odor Class', fontsize=12)
    plt.ylabel('Number of Positive Samples', fontsize=12)
    plt.title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, class_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{int(count)}', ha='center', va='bottom', fontsize=10)
    
    # Performance metrics summary
    plt.subplot(1, 2, 2)
    auc_scores = []
    pr_auc_scores = []
    
    for i in range(len(label_names)):
        if len(np.unique(y_true[:, i])) > 1:
            auc_scores.append(roc_auc_score(y_true[:, i], y_proba[:, i]))
            pr_auc_scores.append(average_precision_score(y_true[:, i], y_proba[:, i]))
        else:
            auc_scores.append(0.0)
            pr_auc_scores.append(0.0)
    
    x_pos = np.arange(len(label_names))
    width = 0.35
    
    plt.bar(x_pos - width/2, auc_scores, width, label='ROC-AUC', color='skyblue')
    plt.bar(x_pos + width/2, pr_auc_scores, width, label='PR-AUC', color='lightcoral')
    
    plt.xlabel('Odor Class', fontsize=12)
    plt.ylabel('AUC Score', fontsize=12)
    plt.title('Performance by Class', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, label_names, rotation=45)
    plt.legend(fontsize=10)
    plt.ylim(0, 1)
    
    # Add value labels
    for i, (auc, pr_auc) in enumerate(zip(auc_scores, pr_auc_scores)):
        plt.text(i - width/2, auc + 0.01, f'{auc:.3f}', ha='center', va='bottom', fontsize=9)
        plt.text(i + width/2, pr_auc + 0.01, f'{pr_auc:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    summary_path = os.path.join(plots_dir, "performance_summary.png")
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   💾 Performance summary saved to: {summary_path}")
    
    print_success(f"All plots saved to: {plots_dir}")

def setup_model_and_tokenizer(config: OdorChemBERTaConfig) -> Tuple:
    """Load and configure ChemBERTa with LoRA for multi-label odor classification"""
    print_step("Setting up LoRA model and tokenizer", f"Checkpoint: {config.MODEL_CHECKPOINT}")
    
    start_time = time.time()
    
    # Load tokenizer
    print("🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_CHECKPOINT)
    tokenizer_time = time.time() - start_time
    
    tokenizer_stats = {
        'Vocabulary size': tokenizer.vocab_size,
        'Model max length': tokenizer.model_max_length,
        'Padding token': tokenizer.pad_token,
        'Load time (s)': f"{tokenizer_time:.2f}"
    }
    print_stats("Tokenizer Information", tokenizer_stats)
    
    # Load base model for multi-label classification
    print("🤖 Loading base model...")
    model_start = time.time()
    
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_CHECKPOINT, 
        num_labels=config.NUM_LABELS,
        problem_type="multi_label_classification"
    )
    
    model_time = time.time() - model_start
    
    # Model statistics before LoRA
    total_params_before = sum(p.numel() for p in model.parameters())
    
    # Configure LoRA
    print("🔧 Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # Sequence classification task
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias=config.LORA_BIAS,
    )
    
    lora_stats = {
        'LoRA rank (r)': config.LORA_R,
        'LoRA alpha': config.LORA_ALPHA,
        'LoRA dropout': config.LORA_DROPOUT,
        'Target modules': ', '.join(config.LORA_TARGET_MODULES),
        'Bias training': config.LORA_BIAS
    }
    print_stats("LoRA Configuration", lora_stats)
    
    # Apply LoRA to model
    print("🚀 Applying LoRA adapters...")
    model = get_peft_model(model, lora_config)
    
    # Model statistics after LoRA
    total_params_after = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    model_stats = {
        'Architecture': str(model.base_model.config.architectures),
        'Base model parameters': f"{total_params_before:,}",
        'Total parameters (with LoRA)': f"{total_params_after:,}",
        'Trainable LoRA parameters': f"{trainable_params:,}",
        'Trainable percentage': f"{(trainable_params/total_params_after)*100:.2f}%",
        'Memory reduction': f"~{((total_params_before-trainable_params)/total_params_before)*100:.1f}%",
        'Hidden size': model.base_model.config.hidden_size,
        'Number of layers': model.base_model.config.num_hidden_layers,
        'Attention heads': model.base_model.config.num_attention_heads,
        'Load time (s)': f"{model_time:.2f}"
    }
    print_stats("LoRA Model Information", model_stats)
    
    # Print classifier head info
    print(f"\n🎯 Classifier head configured for {config.NUM_LABELS} labels:")
    for i, label in enumerate(config.LABEL_NAMES):
        print(f"   Label {i}: {label}")
    
    # Print LoRA details
    print(f"\n⚡ LoRA Efficiency:")
    print(f"   Only training {trainable_params:,} parameters instead of {total_params_before:,}")
    print(f"   That's {(trainable_params/total_params_before)*100:.2f}% of the original model!")
    
    total_time = time.time() - start_time
    print_success(f"LoRA model and tokenizer setup completed in {total_time:.2f} seconds")
    
    return model, tokenizer

def train_chemberta_lora_model(config: OdorChemBERTaConfig = None) -> Tuple:
    """Main LoRA training function for ChemBERTa odor classification"""
    if config is None:
        config = OdorChemBERTaConfig()
    
    print_header("CHEMBERTA LoRA FINE-TUNING FOR ODOR CLASSIFICATION")
    
    # Set random seed for reproducibility
    set_seed(config.RANDOM_SEED)
    
    # Print configuration
    config_info = {
        'Target labels': ', '.join(config.LABEL_NAMES),
        'Number of epochs': config.NUM_EPOCHS,
        'Batch size': config.BATCH_SIZE,
        'Learning rate': config.LEARNING_RATE,
        'Max sequence length': config.MAX_LENGTH,
        'Random seed': config.RANDOM_SEED,
        'Output directory': config.OUTPUT_DIR
    }
    print_stats("Training Configuration", config_info)
    import os
    # Create output directories
    print_step("Creating output directories")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.LOGGING_DIR, exist_ok=True)
    print_success(f"Directories created: {config.OUTPUT_DIR}, {config.LOGGING_DIR}")
    
    # Load datasets
    train_df, test_df, predict_df = load_datasets(config)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Prepare datasets
    train_dataset, test_dataset = prepare_datasets(train_df, test_df, tokenizer, config)
    
    # Training arguments
    print_step("Configuring training arguments")
    
    # Device detection for cross-platform compatibility
    device_info = {
        'CUDA available': torch.cuda.is_available(),
        'CUDA devices': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'MPS available': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
        'Current device': str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'CPU/MPS'
    }
    print_stats("Device Information", device_info)
    
    # Set device for model
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print_success("Using CUDA for training")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print_success("Using Apple Silicon MPS for training")
    else:
        device = torch.device('cpu')
        print_success("Using CPU for training")
    
    # Device configuration for HuggingFace + CUDA compatibility
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        warmup_steps=config.WARMUP_STEPS,
        logging_dir=config.LOGGING_DIR,
        logging_steps=10,  # Log every 10 steps for better training loss tracking
        eval_strategy="epoch",  # Evaluate after each epoch
        save_strategy="no",  # Don't save intermediate checkpoints
        load_best_model_at_end=False,  # Don't load best model (we'll save final)
        dataloader_drop_last=False,
        # Device-specific optimizations
        dataloader_pin_memory=use_cuda,  # Enable for CUDA, disable for MPS
        remove_unused_columns=True,  # Memory optimization
        dataloader_num_workers=0 if use_mps else 2,  # Apple Silicon vs CUDA
        report_to=None,  # Disable wandb
        # Mixed precision for CUDA
        fp16=use_cuda,  # Enable fp16 for CUDA, disable for MPS
        bf16=False,  # Keep disabled for compatibility
        # Learning rate scheduling
        lr_scheduler_type="cosine",  # Cosine decay for better convergence
        # Additional monitoring
        logging_first_step=True,
        log_level="info",
    )
    
    # Calculate training steps
    steps_per_epoch = len(train_dataset) // config.BATCH_SIZE
    total_steps = steps_per_epoch * config.NUM_EPOCHS
    
    training_stats = {
        'Steps per epoch': steps_per_epoch,
        'Total training steps': total_steps,
        'Evaluation frequency': f"Every {training_args.eval_steps} steps",
        'Save frequency': f"Every {training_args.save_steps} steps",
        'Warmup steps': config.WARMUP_STEPS
    }
    print_stats("Training Schedule", training_stats)
    
    # Compute class weights for imbalanced multi-label data
    print_step("Computing class weights for imbalanced data handling")
    train_labels = np.array([item['labels'] for item in train_dataset])
    
    # Calculate positive/negative ratios for each label
    pos_counts = train_labels.sum(axis=0)
    neg_counts = len(train_labels) - pos_counts
    
    # Compute class weights (inverse frequency)
    class_weights = []
    for i, label in enumerate(config.LABEL_NAMES):
        pos_weight = neg_counts[i] / pos_counts[i] if pos_counts[i] > 0 else 1.0
        class_weights.append(pos_weight)
    
    weight_stats = {
        label: f"{weight:.2f}" for label, weight in zip(config.LABEL_NAMES, class_weights)
    }
    print_stats("Class Weights (to handle imbalance)", weight_stats)
    
    # Create trainer with history tracking
    print_step("Initializing trainer with history tracking")
    
    # Create callback for history tracking
    history_callback = TrainingHistoryCallback()
    
    trainer = MultiLabelOdorTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=lambda x: compute_multi_label_metrics(x, config.LABEL_NAMES),
        callbacks=[history_callback],
        class_weights=class_weights,  # Pass class weights for imbalanced data
    )
    print_success("Trainer initialized with BCE loss and history tracking")
    
    # Start training
    print_header("STARTING TRAINING")
    print(f"🚀 Training will run for {config.NUM_EPOCHS} epochs ({total_steps} steps)")
    print(f"📊 Evaluation every {training_args.eval_steps} steps")
    print(f"💾 Model checkpoints saved every {training_args.save_steps} steps")
    
    training_start = time.time()
    
    # Train the model
    trainer.train()
    
    training_time = time.time() - training_start
    print_header("TRAINING COMPLETED")
    print_success(f"Training finished in {training_time/60:.2f} minutes ({training_time:.2f} seconds)")
    
    # Final evaluation
    print_header("FINAL EVALUATION")
    eval_start = time.time()
    eval_results = trainer.evaluate()
    eval_time = time.time() - eval_start
    
    print_success(f"Final evaluation completed in {eval_time:.2f} seconds")
    
    # Log key metrics with better formatting
    print_header("FINAL RESULTS SUMMARY")
    
    # Separate metrics by category
    macro_metrics = {k: v for k, v in eval_results.items() if 'macro' in k}
    micro_metrics = {k: v for k, v in eval_results.items() if 'micro' in k}
    overall_metrics = {k: v for k, v in eval_results.items() if k in ['eval_exact_match', 'eval_hamming_loss', 'eval_loss']}
    
    if macro_metrics:
        print_stats("Macro-Averaged Metrics", macro_metrics)
    if micro_metrics:
        print_stats("Micro-Averaged Metrics", micro_metrics)
    if overall_metrics:
        print_stats("Overall Metrics", overall_metrics)
    
    # Per-label F1 scores
    label_f1s = {k: v for k, v in eval_results.items() if k.endswith('_f1') and not k.endswith(('_macro', '_micro'))}
    if label_f1s:
        print_stats("Per-Label F1 Scores", label_f1s)
    
    # Extract training history from callback
    print_step("Extracting training history for visualization")
    training_history = history_callback.get_history()

    # DEBUG: Print the full training history before plotting
    print("\n[DEBUG] Full training_history before plotting:")
    for k, v in training_history.items():
        print(f"  {k}: {v}")

    # Count non-None values for each metric
    for k, v in training_history.items():
        non_none_count = sum(x is not None for x in v)
        print(f"[DEBUG] {k}: {non_none_count} non-None values out of {len(v)}")

    # Fallback: If all lists are empty or all None, print error and skip plot
    metrics_to_check = ['train_loss', 'eval_loss', 'eval_f1_macro', 'eval_f1_micro']
    all_empty = all(
        not any(x is not None and isinstance(x, (int, float)) for x in training_history.get(metric, []))
        for metric in metrics_to_check
    )
    if all_empty:
        print("[ERROR] No training history data collected. No plot will be generated.")
    else:
        # Filter out None values and ensure alignment (but keep at least some data)
        epochs = training_history['epoch'] if 'epoch' in training_history else []
        train_losses = [loss for loss in training_history['train_loss'] if loss is not None and isinstance(loss, (int, float))]
        eval_losses = [loss for loss in training_history['eval_loss'] if loss is not None and isinstance(loss, (int, float))]
        eval_f1_macro = [f1 for f1 in training_history['eval_f1_macro'] if f1 is not None and isinstance(f1, (int, float))]
        eval_f1_micro = [f1 for f1 in training_history['eval_f1_micro'] if f1 is not None and isinstance(f1, (int, float))]
        learning_rates = [lr for lr in training_history['learning_rate'] if lr is not None and isinstance(lr, (int, float))]

        # Update training_history with clean data
        training_history = {
            'epoch': epochs,
            'train_loss': train_losses,
            'eval_loss': eval_losses,
            'eval_f1_macro': eval_f1_macro,
            'eval_f1_micro': eval_f1_micro,
            'learning_rate': learning_rates
        }

        print(f"📈 Captured {len(train_losses)} training loss points")
        print(f"📈 Captured {len(eval_losses)} validation loss points")
        print(f"📈 Captured {len(eval_f1_macro)} F1 macro points")
        print(f"📈 Captured {len(eval_f1_micro)} F1 micro points")

        print(f"✅ Training completed all {len(epochs)} epochs")

        # Detailed evaluation with predictions for plotting
        print_step("Computing detailed predictions for visualization")
        predictions = trainer.predict(test_dataset)
        probabilities = torch.sigmoid(torch.tensor(predictions.predictions)).numpy()

        # Get true labels
        true_labels = np.array([item['labels'] for item in test_dataset])

        # Create performance plots
        create_performance_plots(
            y_true=true_labels,
            y_proba=probabilities,
            label_names=config.LABEL_NAMES,
            output_dir=config.OUTPUT_DIR,
            training_history=training_history
        )
    
    # Save the final model (simple single checkpoint)
    print_step("Saving final model checkpoint")
    final_model_path = os.path.join(config.OUTPUT_DIR, "final_model")
    
    # Save LoRA model and tokenizer
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print_success(f"Final model saved to: {final_model_path}")
    
    # Print summary
    print_header("TRAINING SUMMARY")
    summary_stats = {
        'Total training time': f"{training_time/60:.2f} minutes",
        'Final macro F1': f"{eval_results.get('eval_f1_macro', 0):.4f}",
        'Final exact match': f"{eval_results.get('eval_exact_match', 0):.4f}",
        'Model saved to': final_model_path,
        'Device used': device_info['Current device']
    }
    print_stats("Training Summary", summary_stats)
    
    return trainer, model, tokenizer, eval_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA fine-tune ChemBERTa for odor classification")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="./chemberta_lora_results", help="Output directory")
    
    args = parser.parse_args()
    
    print_header("COMMAND LINE ARGUMENTS")
    arg_stats = {
        'Epochs': args.epochs,
        'Batch size': args.batch_size,
        'Learning rate': args.learning_rate,
        'LoRA rank': args.lora_r,
        'Output directory': args.output_dir
    }
    print_stats("Arguments", arg_stats)
    
    # Update config with command line arguments
    config = OdorChemBERTaConfig()
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.learning_rate
    config.LORA_R = args.lora_r
    config.RANDOM_SEED = args.seed
    config.OUTPUT_DIR = args.output_dir
    
    # Run LoRA training
    print(f"\n🎯 Starting ChemBERTa LoRA fine-tuning with configuration:")
    print(f"   📁 Output: {config.OUTPUT_DIR}")
    print(f"   🔄 Epochs: {config.NUM_EPOCHS}")
    print(f"   📦 Batch size: {config.BATCH_SIZE}")
    print(f"   📈 Learning rate: {config.LEARNING_RATE}")
    print(f"   🔧 LoRA rank: {config.LORA_R}")
    print(f"   🎲 Random seed: {config.RANDOM_SEED}")
    print(f"   ⚡ Memory efficient: Only training ~2-5% of parameters!")
    print(f"   💡 Inspired by FART approach but for odor (not taste) prediction!")
    
    start_time = time.time()
    trainer, model, tokenizer, results = train_chemberta_lora_model(config)
    total_time = time.time() - start_time
    
    print_header("🎉 SUCCESS!")
    print(f"🎯 ChemBERTa LoRA fine-tuning completed successfully!")
    print(f"⏱️  Total execution time: {total_time/60:.2f} minutes")
    print(f"📁 Results saved to: {config.OUTPUT_DIR}")
    print(f"🏆 Best macro F1 score: {results.get('eval_f1_macro', 0):.4f}")
    print(f"⚡ Memory usage: ~90% less than full fine-tuning!")
    print(f"🧬 FART-inspired ChemBERTa approach for odor classification!")
    print("="*80) 