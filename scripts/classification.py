# -*- coding: utf-8 -*-
"""
This script is used to train and export ML model according to config

Usage:
    python3 ./scripts/classification.py

"""
##################
# Importing libraries
##################
import logging
from pathlib import Path
from utility import parse_config, set_logger
import click
import pandas as pd
import numpy as np
from pycaret.classification import *
import imblearn

#ignore log(0) and divide by 0 warning
np.seterr(divide='ignore')
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


@click.command()
@click.argument("config_file", type=str, default="scripts/config.yml")
def classification(config_file):
    """
    Main function that trains & persists model based on training set

    Args:
        config_file [str]: path to config file

    Returns:
        None
    """
    ##################
    # configure logger
    ##################
    logger = set_logger("./script_logs/classification.log")

    ##################
    # Load config from config file
    ##################
    logger.info(f"Load config from {config_file}")
    config = parse_config(config_file)

    review_data = config["classification"]["lda_review_data_tuned"]
    title_data = config["classification"]["lda_title_data_tuned"]
    predictions_path = Path(config["classification"]["predictions_path"])
    model_classification_path = config["classification"][
        "model_classification_path"]

    logger.info(f"config: {config['classification']}")

    ##################
    # Load and merge data
    ##################
    logger.info(
        f"-------------------Load the processed data-------------------")
    review_data = pd.read_csv(review_data, low_memory=False)
    title_data = pd.read_csv(title_data, low_memory=False)

    data = pd.concat([
        title_data.drop(columns=['Title_sentiment_rating'], axis=1),
        review_data
    ],
                     axis=1)
    data = data.rename(columns={'Review_sentiment_rating': 'sentiment_rating'})
    data = data.dropna(subset=['Title_title', 'Review_content']).reset_index(
        drop=True)

    logger.info(f"shape: {data.shape}")

    logger.info("End Load and merge data")

    ##################
    # Setup
    ##################
    # Setup Pycaret
    logger.info(f"-------------------Setup pycaret-------------------")

    without_pca = setup(
        data=data,
        target='sentiment_rating',
        session_id=42,
        normalize=True,
        transformation=True,
        ignore_features=[
            'Title_title', 'Review_content', 'Title_Dominant_Topic',
            'Title_Perc_Dominant_Topic', 'Review_Dominant_Topic',
            'Review_Perc_Dominant_Topic'
        ],
        use_gpu=True,
        fix_imbalance=True,
        fix_imbalance_method=imblearn.over_sampling.SVMSMOTE(),
        data_split_stratify=True,
        fold_strategy='stratifiedkfold',
        silent=True,
    )

    logger.info(f"End setup pycaret")

    ##################
    # train model
    ##################
    # Train model
    logger.info(f"-------------------Training NB model-------------------")
    model = create_model(estimator='nb')
    predictions = predict_model(model)
    logger.info(f"Metrics:\n {pull(model)}")
    # Finalizing model
    model = finalize_model(model)
    logger.info(f"End training NB model")

    ##################
    # Saving model and predictions
    ##################

    logger.info(f"-------------------Saving model-------------------")
    save_model(model, model_classification_path)
    predictions.to_csv(predictions_path / "predictions.csv", index=False)
    logger.info(f"End saving model")


if __name__ == "__main__":
    classification()
