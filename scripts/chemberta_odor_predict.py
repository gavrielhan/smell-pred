#!/usr/bin/env python3
"""
ChemBERTa LoRA Inference Script for Odor Prediction

Loads a trained ChemBERTa LoRA model and predicts odor labels for a CSV of SMILES.
Supports both single SMILES inference and ensemble inference with synonymous SMILES.
"""
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import os
from rdkit import Chem
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

LABELS = ['sweet', 'floral', 'mint', 'pungent']

def generate_synonymous_smiles(smiles, n=10):
    """Generate n synonymous SMILES for the same molecule using RDKit randomization"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"⚠️  Invalid SMILES: {smiles}")
            return [smiles] * n  # Return original SMILES if invalid
        
        synonymous = []
        for _ in range(n):
            random_smiles = Chem.MolToSmiles(mol, doRandom=True)
            synonymous.append(random_smiles)
        return synonymous
    except Exception as e:
        print(f"⚠️  Error generating synonymous SMILES for {smiles}: {e}")
        return [smiles] * n  # Fallback to original SMILES

def load_model_and_tokenizer(model_path):
    """Load the trained ChemBERTa LoRA model and tokenizer"""
    print(f"Loading model and tokenizer from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        "seyonec/SMILES_tokenized_PubChem_shard00_160k",
        num_labels=len(LABELS),
        problem_type="multi_label_classification"
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    print("✅ Model and tokenizer loaded successfully")
    return model, tokenizer

def predict_odors(model, tokenizer, smiles_list, max_length=256):
    """Run inference on a list of SMILES strings"""
    print(f"🔬 Tokenizing {len(smiles_list)} SMILES...")
    inputs = tokenizer(smiles_list, padding='max_length', truncation=True, 
                      max_length=max_length, return_tensors='pt')

    print("🧠 Running inference...")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.cpu().numpy()
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        preds = (probs > 0.5).astype(int)

    return probs, preds

def ensemble_predict_odors(model, tokenizer, smiles_list, n_ensemble=10, max_length=256, 
                          confidence_threshold=1.0):
    """
    Run ensemble inference using synonymous SMILES for each molecule
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        smiles_list: List of original SMILES
        n_ensemble: Number of synonymous SMILES per molecule
        max_length: Max tokenization length
        confidence_threshold: Threshold for voting (1.0 = unanimous, 0.6 = majority)
    
    Returns:
        final_probs: Averaged probabilities across ensemble
        final_preds: Final predictions based on confidence threshold
        confidence_scores: Confidence scores per label per molecule
    """
    print(f"🎭 Running ensemble inference with {n_ensemble} synonymous SMILES per molecule...")
    
    n_molecules = len(smiles_list)
    final_probs = np.zeros((n_molecules, len(LABELS)))
    final_preds = np.zeros((n_molecules, len(LABELS)), dtype=int)
    confidence_scores = np.zeros((n_molecules, len(LABELS)))
    
    for mol_idx, original_smiles in enumerate(smiles_list):
        if mol_idx % 10 == 0:
            print(f"   Processing molecule {mol_idx + 1}/{n_molecules}...")
        
        # Generate synonymous SMILES
        synonymous_smiles = generate_synonymous_smiles(original_smiles, n_ensemble)
        
        # Run inference on all synonymous SMILES
        ensemble_probs, ensemble_preds = predict_odors(model, tokenizer, synonymous_smiles, max_length)
        
        # Aggregate results
        avg_probs = ensemble_probs.mean(axis=0)  # Average probabilities
        final_probs[mol_idx] = avg_probs
        
        # Compute confidence scores (fraction of agreement)
        for label_idx in range(len(LABELS)):
            votes = ensemble_preds[:, label_idx]
            confidence = votes.mean()  # Fraction of positive votes
            confidence_scores[mol_idx, label_idx] = confidence
            
            # Final prediction based on confidence threshold
            final_preds[mol_idx, label_idx] = 1 if confidence >= confidence_threshold else 0
    
    print("✅ Ensemble inference completed")
    return final_probs, final_preds, confidence_scores

def main():
    parser = argparse.ArgumentParser(description="ChemBERTa LoRA odor prediction")
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to trained model directory')
    parser.add_argument('--input', type=str, required=True, 
                       help='Input CSV with SMILES column')
    parser.add_argument('--output', type=str, required=True, 
                       help='Output CSV for predictions')
    parser.add_argument('--ensemble', action='store_true', 
                       help='Use ensemble inference with synonymous SMILES')
    parser.add_argument('--n_ensemble', type=int, default=10,
                       help='Number of synonymous SMILES for ensemble (default: 10)')
    parser.add_argument('--confidence_threshold', type=float, default=1.0,
                       help='Confidence threshold for ensemble voting (1.0=unanimous, 0.6=majority)')
    parser.add_argument('--find_threshold', action='store_true',
                      help='Find optimal threshold for binarizing probabilities (macro F1, micro F1, exact match)')
    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    # Check if RDKit is available for ensemble mode
    if args.ensemble:
        try:
            import rdkit
            print(f"🎭 Ensemble mode enabled: {args.n_ensemble} SMILES per molecule")
            print(f"🎯 Confidence threshold: {args.confidence_threshold}")
        except ImportError:
            raise ImportError("RDKit is required for ensemble mode. Install with: pip install rdkit")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    # Load input data
    print(f"📊 Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    if 'SMILES' not in df.columns:
        raise ValueError("Input CSV must contain a 'SMILES' column.")
    
    smiles_list = df['SMILES'].tolist()
    print(f"📝 Found {len(smiles_list)} molecules to predict")

    # Run predictions (ensemble or single)
    if args.ensemble:
        probs, preds, confidence_scores = ensemble_predict_odors(
            model, tokenizer, smiles_list, 
            n_ensemble=args.n_ensemble,
            confidence_threshold=args.confidence_threshold
        )
        
        # Add ensemble-specific columns
        for i, label in enumerate(LABELS):
            df[f'pred_{label}'] = preds[:, i]
            df[f'prob_{label}'] = probs[:, i]
            df[f'confidence_{label}'] = confidence_scores[:, i]
        
        # Add overall confidence (average across labels)
        df['confidence_avg'] = confidence_scores.mean(axis=1)
        
        # Print ensemble summary
        print("\n🎭 Ensemble Summary:")
        print(f"   Confidence threshold: {args.confidence_threshold}")
        for i, label in enumerate(LABELS):
            count = preds[:, i].sum()
            avg_conf = confidence_scores[:, i].mean()
            high_conf = (confidence_scores[:, i] >= 0.8).sum()
            print(f"   {label.upper()}: {count}/{len(preds)} predictions ({avg_conf:.3f} avg confidence, {high_conf} high confidence)")
        
    else:
        probs, preds = predict_odors(model, tokenizer, smiles_list)
        
        # Add standard columns
        for i, label in enumerate(LABELS):
            df[f'pred_{label}'] = preds[:, i]
            df[f'prob_{label}'] = probs[:, i]

    # Save output
    df.to_csv(args.output, index=False)
    print(f"💾 Predictions saved to {args.output}")
    
    # Print summary
    print("\n📈 Prediction Summary:")
    for i, label in enumerate(LABELS):
        count = preds[:, i].sum()
        pct = (count / len(preds)) * 100
        print(f"   {label.upper()}: {count}/{len(preds)} ({pct:.1f}%)")
    
    # --- METRICS CALCULATION ---
    # Check if ground truth columns are present
    if all(label in df.columns for label in LABELS):
        print("\n🔬 Calculating metrics (requires ground truth columns in input CSV)...")
        y_true = df[LABELS].values
        y_prob = df[[f'prob_{l}' for l in LABELS]].values

        def compute_metrics_at_threshold(thresh):
            y_pred = (y_prob > thresh).astype(int)
            macro_f1 = f1_score(y_true, y_pred, average='macro')
            micro_f1 = f1_score(y_true, y_pred, average='micro')
            exact_match = (y_pred == y_true).all(axis=1).mean()
            return macro_f1, micro_f1, exact_match

        if args.find_threshold:
            print('\n--- Threshold Sweep (0.00 to 1.00) ---')
            thresholds = np.arange(0.0, 1.01, 0.01)
            macro_f1s, micro_f1s, exacts = [], [], []
            for t in thresholds:
                macro, micro, exact = compute_metrics_at_threshold(t)
                macro_f1s.append(macro)
                micro_f1s.append(micro)
                exacts.append(exact)
            best_macro_idx = int(np.argmax(macro_f1s))
            best_micro_idx = int(np.argmax(micro_f1s))
            best_exact_idx = int(np.argmax(exacts))
            print(f'Best Macro F1: {macro_f1s[best_macro_idx]:.4f} at threshold {thresholds[best_macro_idx]:.2f}')
            print(f'Best Micro F1: {micro_f1s[best_micro_idx]:.4f} at threshold {thresholds[best_micro_idx]:.2f}')
            print(f'Best Exact Match: {exacts[best_exact_idx]:.4f} at threshold {thresholds[best_exact_idx]:.2f}')
            # Use best macro F1 threshold for below metrics printout
            y_pred = (y_prob > thresholds[best_macro_idx]).astype(int)
            print(f'\n--- Metrics at Best Macro F1 Threshold ({thresholds[best_macro_idx]:.2f}) ---')
        else:
            y_pred = df[[f'pred_{l}' for l in LABELS]].values

        # Macro/micro F1, precision, recall
        print('\n--- Macro/Micro Metrics ---')
        print('Macro F1:', f1_score(y_true, y_pred, average='macro'))
        print('Micro F1:', f1_score(y_true, y_pred, average='micro'))
        print('Macro Precision:', precision_score(y_true, y_pred, average='macro'))
        print('Micro Precision:', precision_score(y_true, y_pred, average='micro'))
        print('Macro Recall:', recall_score(y_true, y_pred, average='macro'))
        print('Micro Recall:', recall_score(y_true, y_pred, average='micro'))

        # Per-label metrics
        print('\n--- Per-label Metrics ---')
        for i, label in enumerate(LABELS):
            print(f'Label: {label}')
            print('  F1:', f1_score(y_true[:, i], y_pred[:, i]))
            print('  Precision:', precision_score(y_true[:, i], y_pred[:, i]))
            print('  Recall:', recall_score(y_true[:, i], y_pred[:, i]))
            try:
                print('  ROC-AUC:', roc_auc_score(y_true[:, i], y_prob[:, i]))
            except Exception:
                print('  ROC-AUC: N/A')
            try:
                print('  PR-AUC:', average_precision_score(y_true[:, i], y_prob[:, i]))
            except Exception:
                print('  PR-AUC: N/A')
    else:
        print("\n⚠️  Ground truth columns not found in input CSV. Metrics will not be calculated.")
    
    print("✅ Inference completed successfully!")

if __name__ == "__main__":
    main() 