#!/usr/bin/env python3
"""
FART Fine-tuning for Multi-Label Odor Classification

Adapts the Flavor Analysis and Recognition Transformer (FART) model 
for predicting multiple odor labels: sweet, floral, minty, pungent.

Original FART: Single-label taste classification (sweet, bitter, sour, umami, undefined)
Our adaptation: Multi-label odor classification with sigmoid outputs and BCE loss.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, average_precision_score
)
import os
import argparse
from typing import Dict, List, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OdorFARTConfig:
    """Configuration for FART odor classification fine-tuning"""
    
    # Model configuration
    MODEL_CHECKPOINT = "seyonec/SMILES_tokenized_PubChem_shard00_160k"  # ChemBERTa base
    NUM_LABELS = 4
    LABEL_NAMES = ['sweet', 'floral', 'minty', 'pungent']
    MAX_LENGTH = 512
    
    # Training configuration
    OUTPUT_DIR = "./fart_odor_results"
    LOGGING_DIR = "./fart_odor_logs"
    NUM_EPOCHS = 3
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 500
    
    # Data files
    TRAIN_FILE = "goodscents_train.csv"
    TEST_FILE = "goodscents_test.csv"
    PREDICT_FILE = "bushdid_predict.csv"

class MultiLabelOdorTrainer(Trainer):
    """Custom trainer for multi-label odor classification using BCE loss"""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute BCE loss for multi-label classification"""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        # Use BCE with logits loss for multi-label classification
        loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(logits, labels.float())
        
        return (loss, outputs) if return_outputs else loss

def load_datasets(config: OdorFARTConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, test, and prediction datasets"""
    logger.info("Loading datasets...")
    
    train_df = pd.read_csv(config.TRAIN_FILE)
    test_df = pd.read_csv(config.TEST_FILE)
    predict_df = pd.read_csv(config.PREDICT_FILE)
    
    logger.info(f"Train set: {len(train_df)} molecules")
    logger.info(f"Test set: {len(test_df)} molecules")
    logger.info(f"Prediction set: {len(predict_df)} molecules")
    
    # Show label distribution in training set
    logger.info("Training set label distribution:")
    for label in config.LABEL_NAMES:
        count = train_df[label].sum()
        pct = (count / len(train_df)) * 100
        logger.info(f"  {label.upper()}: {count} molecules ({pct:.1f}%)")
    
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
                    tokenizer, config: OdorFARTConfig) -> Tuple[Dataset, Dataset]:
    """Convert pandas DataFrames to HuggingFace Datasets with tokenization"""
    logger.info("Preparing datasets for training...")
    
    # Convert to HuggingFace Dataset format
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Tokenize SMILES
    train_dataset = train_dataset.map(
        lambda x: tokenize_smiles(x, tokenizer, config.MAX_LENGTH), 
        batched=True
    )
    test_dataset = test_dataset.map(
        lambda x: tokenize_smiles(x, tokenizer, config.MAX_LENGTH), 
        batched=True
    )
    
    # Prepare multi-label targets
    def format_labels(examples):
        """Convert multi-label columns to tensor format"""
        labels = []
        for i in range(len(examples[config.LABEL_NAMES[0]])):
            label_vector = [examples[label][i] for label in config.LABEL_NAMES]
            labels.append(label_vector)
        return {"labels": labels}
    
    train_dataset = train_dataset.map(format_labels, batched=True)
    test_dataset = test_dataset.map(format_labels, batched=True)
    
    # Set format for PyTorch
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    logger.info(f"✅ Prepared datasets with {len(train_dataset)} train and {len(test_dataset)} test samples")
    
    return train_dataset, test_dataset

def compute_multi_label_metrics(eval_pred, label_names: List[str]) -> Dict:
    """Compute comprehensive metrics for multi-label classification"""
    logits, labels = eval_pred
    
    # Convert logits to probabilities using sigmoid
    predictions_proba = torch.sigmoid(torch.tensor(logits)).numpy()
    
    # Convert to binary predictions (threshold = 0.5)
    predictions_binary = (predictions_proba > 0.5).astype(int)
    
    results = {}
    
    # Per-label metrics
    for i, label_name in enumerate(label_names):
        y_true = labels[:, i]
        y_pred = predictions_binary[:, i]
        y_proba = predictions_proba[:, i]
        
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
    
    return results

def setup_model_and_tokenizer(config: OdorFARTConfig) -> Tuple:
    """Load and configure the FART model and tokenizer for multi-label classification"""
    logger.info(f"Loading FART model: {config.MODEL_CHECKPOINT}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_CHECKPOINT)
    logger.info(f"✅ Loaded tokenizer (vocab_size: {tokenizer.vocab_size})")
    
    # Load model for multi-label classification
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_CHECKPOINT, 
        num_labels=config.NUM_LABELS,
        problem_type="multi_label_classification"
    )
    
    logger.info(f"✅ Loaded model for {config.NUM_LABELS} labels: {config.LABEL_NAMES}")
    logger.info(f"Model architecture: {model.config.architectures}")
    
    return model, tokenizer

def train_fart_odor_model(config: OdorFARTConfig = None) -> Tuple:
    """Main training function for FART odor classification"""
    if config is None:
        config = OdorFARTConfig()
    
    logger.info("🚀 Starting FART fine-tuning for odor classification")
    logger.info(f"Target labels: {config.LABEL_NAMES}")
    
    # Create output directories
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.LOGGING_DIR, exist_ok=True)
    
    # Load datasets
    train_df, test_df, predict_df = load_datasets(config)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Prepare datasets
    train_dataset, test_dataset = prepare_datasets(train_df, test_df, tokenizer, config)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        warmup_steps=config.WARMUP_STEPS,
        logging_dir=config.LOGGING_DIR,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=3,
        dataloader_drop_last=False,
        report_to=None,  # Disable wandb for now
    )
    
    # Create trainer
    trainer = MultiLabelOdorTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=lambda x: compute_multi_label_metrics(x, config.LABEL_NAMES),
    )
    
    logger.info("🔥 Starting training...")
    
    # Train the model
    trainer.train()
    
    logger.info("✅ Training completed!")
    
    # Final evaluation
    logger.info("📊 Running final evaluation...")
    eval_results = trainer.evaluate()
    
    # Log key metrics
    logger.info("=== Final Evaluation Results ===")
    for metric, value in eval_results.items():
        if 'f1' in metric or 'exact_match' in metric:
            logger.info(f"{metric}: {value:.4f}")
    
    # Save the final model
    trainer.save_model(os.path.join(config.OUTPUT_DIR, "final_model"))
    tokenizer.save_pretrained(os.path.join(config.OUTPUT_DIR, "final_model"))
    
    logger.info(f"💾 Model saved to {config.OUTPUT_DIR}/final_model")
    
    return trainer, model, tokenizer, eval_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune FART for odor classification")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="./fart_odor_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config = OdorFARTConfig()
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.learning_rate
    config.OUTPUT_DIR = args.output_dir
    
    # Run training
    trainer, model, tokenizer, results = train_fart_odor_model(config)
    
    print("\n🎯 FART fine-tuning completed successfully!")
    print(f"📁 Results saved to: {config.OUTPUT_DIR}") 