import os
import sys
import logging
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem,Draw
from src.utils import ensure_dir,setup_logger
from keras.models import load_model
from PIL import Image
import tensorflow as tf
import numpy as np

def load_model(model_path='cnn_tki_classifier.h5'):
    """Load a trained CNN model from model_path."""
    if not os.path.exists(model_path):
        logging.error(f"Model file {model_path} not found.")
        return None
    model = tf.keras.models.load_model(model_path)
    logging.info(f"Model loaded from {model_path}")
    return model

def predict_smiles(smiles,model):
    """Generate 224x224 image from SMILES and predict activity with model.
    whether it's active (1) or inactive (0).
    Returns a float (probability of being active)."""

    mol=Chem.MolFromSmiles(smiles)
    if not mol:
        logging.error(f"Incorrect SMILES string: {smiles}")
        return None
    img = Draw.MolToImage(mol,size=(224,224))
    img_arr= np.array(img)/255.0
    img_arr=np.expand_dims(img_arr,axis=0)

    prob=model.predict(img_arr)[0][0]
    logging.info(f"Predicted probability for SMILES {smiles}: {prob}")
    return prob

if __name__ == "__main__":
    setup_logger()
    model = load_model()
    if model:
        test_smiles = "CCO"
        probability = predict_smiles(test_smiles, model)
        if probability is not None:
            logging.info(f"Predicted probability of activity: {probability:.4f}")
