import pandas as pd
import os

labels = ['pungent', 'sweet', 'floral', 'mint']
splits = ['train', 'val', 'test']

os.makedirs('data', exist_ok=True)

def extract(row):
    odor_set = set(str(row).replace(' ', '').split(','))
    return [int(l in odor_set) for l in labels]

for split in splits:
    df = pd.read_excel(f'Mol-PECO/data/pyrfume_{split}.xlsx')
    new_rows = []
    for i, row in df.iterrows():
        vals = extract(row['Odor'])
        if any(vals):
            new_rows.append([row['SMILES']] + vals)
    out_df = pd.DataFrame(new_rows, columns=['SMILES'] + labels)
    out_df.to_csv(f'data/pyrfume_{split}_4odors.csv', index=False)
print('Conversion complete. Files saved to data/.') 