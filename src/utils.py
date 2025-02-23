#src/utils.py

import os
import sys

import logging
import logging.config

def setup_logger(log_name="app.log", log_level=logging.INFO):
    """Sets up logger which writes to console and file."""
    logger = logging.getLogger()
    logger.setLevel(log_level)


    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Log to file
        fh = logging.FileHandler(log_name)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Log to console
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)


def ensure_dir(directory):
    """Ensure that the directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)