import pandas as pd
import logging
import numpy as np
import os

def ism_csv(ism_file,label,output_csv):
    """Converts an ISM file (one SMILES per line) to a CSV with columns SMILES, Activity."""

    logging.info(f"Reading ISM file {ism_file}")

    try:
        with open(ism_file,"r") as f:
            smiles_list = [line.strip() for line in f if line.strip()]
            df = pd.DataFrame(smiles_list,columns=["SMILES"])
            df["Activity"] = label
            df.to_csv(output_csv,index=False)
            logging.info(f"ISM file {ism_file} converted to CSV {output_csv}")
    except Exception as e:
        logging.error(f"Error converting ISM file {ism_file} to CSV: {e}")
        return False
    

def merge_datasets(actives_csv,decoys_csv,merged_csv):
    """ Merges active and decoy datasets into a single CSV with SMILES and Activity."""
    logging.info("Merging active and decoy datasets")
    try:
        actives_df = pd.read_csv(actives_csv)
        decoys_df = pd.read_csv(decoys_csv)
        merged_df = pd.concat([actives_df,decoys_df],ignore_index=True)
        merged_df.to_csv(merged_csv,index=False)
        logging.info(f"Merged dataset saved to {merged_csv},Merged dataset has {len(merged_df)} molecules.")
        return True
    except Exception as e:
        logging.error(f"Error merging datasets: {e}")
        return False
    