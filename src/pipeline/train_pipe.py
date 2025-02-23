# src/pipeline/train_pipe.py
import logging
from src.utils import setup_logger
from src.components.data_ingestion import ism_csv, merge_datasets
from src.components.data_transformation import generate_images_from_csv
from src.components.model_trainer import train_cnn

def run_full_training_pipeline(
    actives_ism="actives_final.ism",
    decoys_ism="decoys_final.ism",
    actives_csv="actives.csv",
    decoys_csv="decoys.csv",
    merged_csv="molecules.csv",
    dataset_dir="dataset",
    epochs=5
):
    setup_logger()
    logging.info("Starting full training pipeline.")
    
    # 1. Convert decoys from ISM to CSV (label=0)
    ism_csv(decoys_ism, 0, decoys_csv)
    ism_csv(actives_ism, 1, actives_csv)
    # 2. Merge actives & decoys into one CSV
    merge_datasets(actives_csv, decoys_csv, merged_csv)
    
    # 3. Generate images
    generate_images_from_csv(merged_csv, output_base=dataset_dir)
    
    # 4. Train the CNN
    train_cnn(dataset_dir=dataset_dir, epochs=epochs)
    logging.info("Training pipeline completed.")

if __name__ == "__main__":
    run_full_training_pipeline()
