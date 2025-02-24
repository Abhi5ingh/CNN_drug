import os
import logging
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem,Draw 
from src.utils import ensure_dir

def generate_images(smiles,save_path,size=(224,224)):
    """Generate and save a molecule image from a SMILES string.
    returs True if successfull."""

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            img = Draw.MolToImage(mol,size=size)
            img.save(save_path)
            return True
        else:
            logging.error(f"Incorrect SMILES string: {smiles}")
            return False
    except Exception as e:
        logging.error(f"Error generating image for SMILES {smiles}: {e}")
        return False
    
def generate_images_from_csv(csv_file,output_base="dataset"):
    """Reads SMILES and activity from a CSV and generates images in subgfolders of output_base."""
    if os.path.exists(output_base) and os.listdir(output_base):

        logging.info(f"'{output_base}' is not empty. Skipping image generation.")
        return True
    else:

        logging.info(f"generating images from {csv_file}")
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            logging.error(f"Error reading CSV {csv_file}")
            return False
        if df.empty:
            logging.error(f"Empty CSV {csv_file}")
            return False
        
        active_dir = os.path.join(output_base,"1_active")
        inactive_dir = os.path.join(output_base,"0_inactive")

        ensure_dir(active_dir)
        ensure_dir(inactive_dir)

        total = len(df)
        success = 0
        for i,row in df.iterrows():
            smiles = row.get("SMILES")
            label = row.get("Activity")
            
            if pd.isna(smiles) or pd.isna(label):
                logging.error(f"Missing SMILES or Activity in row {i}")
                continue
            folder = active_dir if label ==1 else inactive_dir
            file_path = os.path.join(folder,f"mol_{i}.png")
            if generate_images(smiles,file_path):
                success += 1

        logging.info(f"Generated {success} images out of {total}")
        return True