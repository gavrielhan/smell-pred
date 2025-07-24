import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io
import matplotlib.pyplot as plt
import joblib
from sklearn.multioutput import MultiOutputClassifier

# --- CONFIG ---
DATA_PATH = "data/pyrfume_test_4odors.csv"
MODEL_PATH = "ml_odor_results/lightgbm_model.txt"  # or retrain if not present
LABELS = ['sweet', 'floral', 'mint', 'pungent']
N_EXAMPLES = 2
TOP_N_ATOMS = 2
IMG_SIZE = (300, 300)
FP_RADIUS = 2
FP_NBITS = 2048

# --- LOAD MOLECULES ---
df = pd.read_csv(DATA_PATH)
examples = pd.concat([
    df.iloc[[0]],  # first molecule
    df.iloc[[126]] # longest molecule
], ignore_index=True)

# --- FEATURE EXTRACTION ---
def smiles_to_fp_and_bitinfo(smiles):
    mol = Chem.MolFromSmiles(smiles)
    bitInfo = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_NBITS, bitInfo=bitInfo)
    arr = np.zeros((1, FP_NBITS), dtype=int)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr[0])
    return mol, arr, bitInfo

# --- LOAD OR TRAIN LIGHTGBM MODEL ---
def load_or_train_model():
    import os
    pkl_path = MODEL_PATH.replace('.txt', '.pkl')
    if os.path.exists(pkl_path):
        model = joblib.load(pkl_path)
        return model
    # Otherwise, train a new model on pyrfume data
    train_df = pd.read_csv("data/pyrfume_train_4odors.csv")
    X = []
    for smi in train_df['SMILES']:
        _, arr, _ = smiles_to_fp_and_bitinfo(smi)
        X.append(arr[0])
    X = np.array(X)
    y = train_df[LABELS].values
    model = MultiOutputClassifier(lgb.LGBMClassifier(n_estimators=200))
    model.fit(X, y)
    joblib.dump(model, pkl_path)
    return model

# --- GAUSSIAN HALO HELPER ---
def draw_gaussian_halo(ax, x, y, color, magnitude, radius=40, alpha_max=0.5):
    size = radius * 2
    grid = np.linspace(-radius, radius, size)
    X, Y = np.meshgrid(grid, grid)
    sigma = radius / 2.5
    Z = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    Z = Z / Z.max()
    rgba = np.zeros((size, size, 4))
    if color == 'green':
        rgba[..., 1] = 1.0
    else:
        rgba[..., 0] = 1.0
    rgba[..., 3] = alpha_max * abs(magnitude) * Z
    ax.imshow(rgba, extent=(x-radius, x+radius, y-radius, y+radius), origin='lower')

# --- DRAW MOLECULE WITH SHAP HALOS ---
def draw_molecule_with_shap_halos_full(mol, atom_scores_by_label, smiles, true_labels, pred_labels, ax=None, img_size=(300, 300)):
    drawer = rdMolDraw2D.MolDraw2DCairo(img_size[0], img_size[1])
    options = drawer.drawOptions()
    options.useBWAtomPalette()
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    img_bytes = drawer.GetDrawingText()
    img = Image.open(io.BytesIO(img_bytes))
    atom_coords = [drawer.GetDrawCoords(i) for i in range(mol.GetNumAtoms())]
    if ax is None:
        fig, ax = plt.subplots(figsize=(img_size[0]/100, img_size[1]/100), dpi=100)
    ax.imshow(img)
    ax.set_xlim(0, img.size[0])
    ax.set_ylim(img.size[1], 0)
    ax.axis('off')
    # For each atom, sum SHAPs across all labels
    atom_sums = [sum(scores) for scores in atom_scores_by_label]
    max_score = max(abs(x) for x in atom_sums) if atom_sums else 1
    for i, score in enumerate(atom_sums):
        color = 'green' if score >= 0 else 'red'
        magnitude = abs(score) / max_score if max_score > 0 else 0.5
        x, y = atom_coords[i]
        draw_gaussian_halo(ax, x, y, color, magnitude)
    true_str = ','.join([l for l, t in zip(LABELS, true_labels) if t])
    pred_str = ','.join(pred_labels) if pred_labels else 'None'
    ax.set_title(f"True: {true_str}\nPred: {pred_str}\n{smiles}", fontsize=12)

# --- MAIN ---
model = load_or_train_model()

fig, axes = plt.subplots(1, N_EXAMPLES, figsize=(5*N_EXAMPLES, 5))
if N_EXAMPLES == 1:
    axes = [axes]
for i, (idx, row) in enumerate(examples.iterrows()):
    smiles = row['SMILES']
    mol, arr, bitInfo = smiles_to_fp_and_bitinfo(smiles)
    pred_probs = np.array([est.predict_proba(arr)[0, 1] for est in model.estimators_])
    pred_labels = [LABELS[j] for j, p in enumerate(pred_probs) if p > 0.5]
    true_labels = [int(row[l]) for l in LABELS]
    atom_scores_by_label = [[0.0 for _ in LABELS] for _ in range(mol.GetNumAtoms())]
    for label_idx, est in enumerate(model.estimators_):
        explainer = shap.TreeExplainer(est)
        shap_vals = explainer.shap_values(arr)[0]
        for bit_id, atom_tuples in bitInfo.items():
            shap_val = shap_vals[bit_id]
            for atom_idx, _ in atom_tuples:
                atom_scores_by_label[atom_idx][label_idx] += shap_val
    draw_molecule_with_shap_halos_full(mol, atom_scores_by_label, smiles, true_labels, pred_labels, ax=axes[i], img_size=IMG_SIZE)
plt.suptitle("Tree Model SHAP: Full Mask (Green=Support, Red=Oppose, Sum over Labels)", fontsize=16)
plt.tight_layout()
plt.show() 