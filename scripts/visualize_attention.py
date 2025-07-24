import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
import matplotlib.pyplot as plt
import os
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io
import matplotlib.patches as patches

# --- CONFIG ---
MODEL_DIR = "chemberta_lora_results/final_model"
CSV_PATH = "data/bushdid_predict.csv"
N_EXAMPLES = 2
TOP_N_ATOMS = 2
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
# Use the first molecule and the longest molecule (index 126)
examples = pd.concat([
    df.iloc[[0]],  # first molecule
    df.iloc[[126]] # longest molecule
], ignore_index=True)

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

# Helper: Draw a colored Gaussian halo at (x, y)
def draw_gaussian_halo(ax, x, y, color, magnitude, radius=40, alpha_max=0.5):
    # Create a 2D Gaussian as an image
    size = radius * 2
    grid = np.linspace(-radius, radius, size)
    X, Y = np.meshgrid(grid, grid)
    sigma = radius / 2.5
    Z = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    Z = Z / Z.max()  # Normalize
    # Color and alpha scaling
    if color == 'green':
        rgba = np.zeros((size, size, 4))
        rgba[..., 1] = 1.0  # Green channel
    else:
        rgba = np.zeros((size, size, 4))
        rgba[..., 0] = 1.0  # Red channel
    # Alpha: scale by magnitude and Z
    rgba[..., 3] = alpha_max * abs(magnitude) * Z
    # Overlay on ax
    ax.imshow(rgba, extent=(x-radius, x+radius, y-radius, y+radius), origin='lower')

# --- DRAW MOLECULE WITH GAUSSIAN HALOS (TOP 2 ONLY, BW IMAGE) ---
def draw_molecule_with_gaussian_halos(mol, atom_scores, smiles, ax=None, img_size=(300, 300), top_n=2):
    # Draw molecule in black and white
    drawer = rdMolDraw2D.MolDraw2DCairo(img_size[0], img_size[1])
    options = drawer.drawOptions()
    options.useBWAtomPalette()
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    img_bytes = drawer.GetDrawingText()
    img = Image.open(io.BytesIO(img_bytes))
    # Get atom pixel coordinates
    atom_coords = [drawer.GetDrawCoords(i) for i in range(mol.GetNumAtoms())]
    if ax is None:
        fig, ax = plt.subplots(figsize=(img_size[0]/100, img_size[1]/100), dpi=100)
    ax.imshow(img)
    ax.set_xlim(0, img.size[0])
    ax.set_ylim(img.size[1], 0)
    ax.axis('off')
    # Only plot top_n atoms by |importance|
    if not atom_scores or len(atom_scores) == 0:
        return
    top_indices = np.argsort(np.abs(atom_scores))[-top_n:]
    max_score = max(abs(x) for x in atom_scores) if atom_scores else 1
    for i in top_indices:
        x, y = atom_coords[i]
        score = atom_scores[i]
        color = 'green' if score >= 0 else 'red'
        # Normalize magnitude for alpha
        magnitude = abs(score) / max_score if max_score > 0 else 0.5
        draw_gaussian_halo(ax, x, y, color, magnitude)
    ax.set_title(smiles)

# --- MAIN VISUALIZATION ---
fig, axes = plt.subplots(1, N_EXAMPLES, figsize=(5*N_EXAMPLES, 5))
if N_EXAMPLES == 1:
    axes = [axes]
for i, (idx, row) in enumerate(examples.iterrows()):
    smiles = row['IsomericSMILES']
    print(f"Processing molecule: {smiles}")
    token_importance, inputs = get_token_attention(smiles)
    mol, atom_scores = map_token_importance_to_atoms(smiles, token_importance, inputs)
    # Predict odors
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.cpu().numpy()[0]
        probs = 1 / (1 + np.exp(-logits))
        pred_labels = [LABELS[j] for j, p in enumerate(probs) if p > 0.5]
        pred_label_str = ', '.join(pred_labels) if pred_labels else 'None'
    draw_molecule_with_gaussian_halos(mol, atom_scores, smiles, ax=axes[i], top_n=TOP_N_ATOMS)
    # Add predicted label(s) above the molecule
    axes[i].set_title(f"Predicted: {pred_label_str}\n{smiles}", fontsize=12)
plt.suptitle("Attention Visualization: Top 2 Gaussian Halos (Green=Support, Red=Oppose)", fontsize=16)
plt.tight_layout()
plt.show()
