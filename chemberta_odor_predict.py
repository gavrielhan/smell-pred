#!/usr/bin/env python3
"""
ChemBERTa Odor Prediction Script

Uses the fine-tuned ChemBERTa model to predict odor labels for new molecules.
Supports both single molecule predictions and batch predictions from CSV files.

Inspired by FART (Flavor Analysis and Recognition Transformer) approach.
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
import os
from typing import List, Dict, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OdorPredictor:
    """Wrapper class for making odor predictions with fine-tuned ChemBERTa model"""
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the predictor with a trained model
        
        Args:
            model_path: Path to the fine-tuned model directory
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
        """
        self.model_path = model_path
        self.label_names = ['sweet', 'floral', 'minty', 'pungent']
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self._load_model()
    
    def _load_model(self):
        """Load the fine-tuned model and tokenizer"""
        logger.info(f"Loading model from: {self.model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("✅ Model and tokenizer loaded successfully")
            logger.info(f"Model has {self.model.config.num_labels} output labels")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict_single(self, smiles: str, return_probabilities: bool = True) -> Dict:
        """
        Predict odor labels for a single SMILES string
        
        Args:
            smiles: SMILES string representing the molecule
            return_probabilities: Whether to return probability scores
            
        Returns:
            Dictionary with predictions and optionally probabilities
        """
        # Tokenize the SMILES
        inputs = self.tokenizer(
            smiles, 
            padding=True, 
            truncation=True, 
            max_length=512,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Convert to probabilities using sigmoid
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]
            
            # Convert to binary predictions (threshold = 0.5)
            predictions = (probabilities > 0.5).astype(int)
        
        # Format results
        result = {
            'smiles': smiles,
            'predictions': {label: int(pred) for label, pred in zip(self.label_names, predictions)}
        }
        
        if return_probabilities:
            result['probabilities'] = {label: float(prob) for label, prob in zip(self.label_names, probabilities)}
        
        return result
    
    def predict_batch(self, smiles_list: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Predict odor labels for a batch of SMILES strings
        
        Args:
            smiles_list: List of SMILES strings
            batch_size: Batch size for processing
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        logger.info(f"Processing {len(smiles_list)} molecules in batches of {batch_size}")
        
        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i:i+batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_smiles,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Convert to probabilities using sigmoid
                probabilities = torch.sigmoid(logits).cpu().numpy()
                
                # Convert to binary predictions (threshold = 0.5)
                predictions = (probabilities > 0.5).astype(int)
            
            # Format results for this batch
            for j, smiles in enumerate(batch_smiles):
                result = {
                    'smiles': smiles,
                    'predictions': {label: int(pred) for label, pred in zip(self.label_names, predictions[j])},
                    'probabilities': {label: float(prob) for label, prob in zip(self.label_names, probabilities[j])}
                }
                results.append(result)
        
        logger.info(f"✅ Completed predictions for {len(results)} molecules")
        return results
    
    def predict_from_csv(self, input_file: str, output_file: str = None, 
                        smiles_column: str = "IsomericSMILES", 
                        batch_size: int = 32) -> pd.DataFrame:
        """
        Predict odor labels for molecules in a CSV file
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to save results (optional)
            smiles_column: Name of the column containing SMILES
            batch_size: Batch size for processing
            
        Returns:
            DataFrame with predictions added
        """
        logger.info(f"Loading molecules from: {input_file}")
        
        # Load the CSV file
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} molecules")
        
        if smiles_column not in df.columns:
            raise ValueError(f"Column '{smiles_column}' not found in {input_file}")
        
        # Get SMILES strings
        smiles_list = df[smiles_column].dropna().tolist()
        logger.info(f"Found {len(smiles_list)} valid SMILES strings")
        
        # Make predictions
        predictions = self.predict_batch(smiles_list, batch_size)
        
        # Add predictions to dataframe
        for i, pred in enumerate(predictions):
            row_idx = df[smiles_column].dropna().index[i]
            
            # Add binary predictions
            for label in self.label_names:
                df.loc[row_idx, f'pred_{label}'] = pred['predictions'][label]
                df.loc[row_idx, f'prob_{label}'] = pred['probabilities'][label]
        
        # Add summary columns
        pred_cols = [f'pred_{label}' for label in self.label_names]
        df['num_predicted_odors'] = df[pred_cols].sum(axis=1)
        df['max_probability'] = df[[f'prob_{label}' for label in self.label_names]].max(axis=1)
        df['predicted_odors'] = df.apply(
            lambda row: [label for label in self.label_names if row[f'pred_{label}'] == 1], 
            axis=1
        )
        
        # Save results if output file specified
        if output_file:
            df.to_csv(output_file, index=False)
            logger.info(f"💾 Results saved to: {output_file}")
            
            # Print summary
            logger.info("\n=== Prediction Summary ===")
            for label in self.label_names:
                count = df[f'pred_{label}'].sum()
                pct = (count / len(df)) * 100
                logger.info(f"{label.upper()}: {count} molecules ({pct:.1f}%)")
            
            multi_label_count = (df['num_predicted_odors'] > 1).sum()
            logger.info(f"Multi-label molecules: {multi_label_count} ({(multi_label_count/len(df)*100):.1f}%)")
        
        return df

def main():
    parser = argparse.ArgumentParser(description="Predict odor labels using fine-tuned ChemBERTa model")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to the fine-tuned model directory")
    parser.add_argument("--input", type=str, 
                       help="Input CSV file with SMILES (for batch prediction)")
    parser.add_argument("--output", type=str, 
                       help="Output CSV file for results")
    parser.add_argument("--smiles", type=str, 
                       help="Single SMILES string for prediction")
    parser.add_argument("--smiles_column", type=str, default="IsomericSMILES",
                       help="Name of SMILES column in CSV")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for predictions")
    parser.add_argument("--device", type=str, 
                       help="Device to use (cpu/cuda)")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = OdorPredictor(args.model_path, args.device)
    
    if args.smiles:
        # Single molecule prediction
        logger.info(f"Predicting odor for SMILES: {args.smiles}")
        result = predictor.predict_single(args.smiles)
        
        print("\n=== Prediction Results ===")
        print(f"SMILES: {result['smiles']}")
        print("Predicted odors:")
        for label, pred in result['predictions'].items():
            prob = result['probabilities'][label]
            status = "✅" if pred else "❌"
            print(f"  {status} {label.upper()}: {prob:.3f}")
        
        predicted_labels = [label for label, pred in result['predictions'].items() if pred]
        if predicted_labels:
            print(f"\nSummary: {', '.join(predicted_labels)}")
        else:
            print("\nSummary: No odor predicted")
    
    elif args.input:
        # Batch prediction from CSV
        if not args.output:
            # Generate output filename
            base_name = os.path.splitext(args.input)[0]
            args.output = f"{base_name}_with_odor_predictions.csv"
        
        results_df = predictor.predict_from_csv(
            args.input, args.output, args.smiles_column, args.batch_size
        )
        
        print(f"\n✅ Predictions completed!")
        print(f"📁 Results saved to: {args.output}")
    
    else:
        parser.print_help()
        print("\nError: Please provide either --smiles for single prediction or --input for batch prediction")

if __name__ == "__main__":
    main() 