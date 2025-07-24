import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIG ---
MODEL_DIR = "chemberta_lora_results/final_model"
CSV_PATH = "data/bushdid_predict.csv"
N_EXAMPLES = 2
TOP_N_ATOMS = 3
LABELS = ['sweet', 'floral', 'minty', 'pungent']

# --- LOAD MODEL & TOKENIZER ---
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
# Load base model and then LoRA adapter
base_model = AutoModelForSequenceClassification.from_pretrained(
    "seyonec/SMILES_tokenized_PubChem_shard00_160k",
    num_labels=len(LABELS),
    problem_type="multi_label_classification"
)
model = PeftModel.from_pretrained(base_model, MODEL_DIR)
model.eval()

# --- LOAD MOLECULES ---
df = pd.read_csv(CSV_PATH)
examples = df.iloc[:N_EXAMPLES]

# --- ATTENTION VISUALIZATION ---
def get_token_attention(smiles):
    inputs = tokenizer(smiles, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    # Get attentions: tuple of (layer, batch, head, seq, seq)
    # We'll use the last layer, average over heads, and sum attention to each input token
    attn = outputs.attentions[-1]  # (batch, heads, seq, seq)
    attn = attn[0]  # (heads, seq, seq)
    attn_mean = attn.mean(0)  # (seq, seq)
    # For each input token, sum attention received from all other tokens
    token_importance = attn_mean.sum(0).cpu().numpy()  # (seq,)
    return token_importance, inputs

# --- MAP TOKENS TO ATOMS (APPROXIMATE) ---
def map_token_importance_to_atoms(smiles, token_importance, inputs):
    # Get mapping from tokens to SMILES chars
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    # Map each token to a SMILES char index (approximate)
    smiles_chars = list(smiles)
    atom_importance = np.zeros(len(smiles_chars))
    char_pointer = 0
    for i, tok in enumerate(tokens):
        if tok in [tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token, tokenizer.mask_token, None]:
            continue
        # Remove special chars from token (e.g., Ġ)
        clean_tok = tok.replace('Ġ', '').replace('▁', '')
        if not clean_tok:
            continue
        # Find where this token appears in the SMILES string
        idx = smiles.find(clean_tok, char_pointer)
        if idx != -1:
            for j in range(idx, idx+len(clean_tok)):
                if j < len(atom_importance):
                    atom_importance[j] += token_importance[i]
            char_pointer = idx + len(clean_tok)
    # Now, map SMILES chars to atoms using RDKit
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    atom_map = {i: [] for i in range(mol.GetNumAtoms())}
    # Map each atom index to the SMILES char indices that represent it
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        idx = atom.GetIdx()
        # Find all occurrences of the atom symbol in the SMILES string
        for i, c in enumerate(smiles_chars):
            if c == sym:
                atom_map[idx].append(i)
    # Aggregate importance for each atom
    atom_scores = []
    for idx in range(mol.GetNumAtoms()):
        if atom_map[idx]:
            score = np.sum([atom_importance[i] for i in atom_map[idx]])
        else:
            score = 0.0
        atom_scores.append(score)
    return mol, atom_scores

# --- DRAW MOLECULE WITH HIGHLIGHTED ATOMS ---
def draw_molecule_with_attention(mol, atom_scores, smiles, top_n=3, ax=None):
    if mol is None or atom_scores is None:
        print(f"Could not parse molecule: {smiles}")
        return
    # Get top N atom indices
    top_atoms = np.argsort(atom_scores)[-top_n:]
    # Normalize scores for color intensity
    max_score = max(atom_scores) if max(atom_scores) > 0 else 1
    colors = {idx: (1, 0, 0) for idx in top_atoms}  # Red for top atoms
    # Draw molecule
    img = Draw.MolToImage(mol, size=(300, 300), highlightAtoms=top_atoms.tolist(), highlightAtomColors=colors)
    if ax is not None:
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(smiles)
    else:
        img.show()

# --- MAIN VISUALIZATION ---
fig, axes = plt.subplots(1, N_EXAMPLES, figsize=(5*N_EXAMPLES, 5))
if N_EXAMPLES == 1:
    axes = [axes]
for i, (idx, row) in enumerate(examples.iterrows()):
    smiles = row['IsomericSMILES']
    print(f"Processing molecule: {smiles}")
    token_importance, inputs = get_token_attention(smiles)
    mol, atom_scores = map_token_importance_to_atoms(smiles, token_importance, inputs)
    draw_molecule_with_attention(mol, atom_scores, smiles, top_n=TOP_N_ATOMS, ax=axes[i])
plt.suptitle("Attention Visualization: Top 2-3 Atoms Highlighted (Red Halo)", fontsize=16)
plt.tight_layout()
plt.show()

# --- ROC-AUC EXPLANATION ---
print("\n--- ROC-AUC for 'sweet' class (most prevalent) is lowest: Why? ---")
print("""
1. ROC-AUC measures the model's ability to rank positives above negatives. If a class is very common, the model may predict high probabilities for most samples, making it hard to distinguish true positives from false positives.
2. The model may learn to always predict 'sweet', reducing its discrimination power (high recall, low precision).
3. Imbalanced classes can lead to poor ROC-AUC for the majority class, especially if the negatives are rare and hard to separate.
4. Label noise or ambiguity in 'sweet' can also lower ROC-AUC.
5. In multi-label settings, co-occurrence with other labels can confuse the model for the most common class.
""") 