# -*- coding: utf-8 -*-
"""
This script is used to do prediction based on trained model

Usage:
    python3 ./scripts/predict.py

"""
import logging
from pathlib import Path
import click
import pandas as pd
from pycaret.nlp import *
from pycaret.classification import load_model, predict_model

from utility import parse_config, set_logger
import utility

# NLTK
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet, stopwords


@click.command()
@click.argument("config_file", type=str, default="scripts/config.yml")
def predict(config_file):
    """
    Main function that runs predictions

    Args:
        config_file [str]: path to config file

    Returns:
        None
    """
    ##################
    # configure logger
    ##################
    logger = set_logger("./log/predict.log")

    ##################
    # Load config from config file
    ##################
    logger.info(f"Load config from {config_file}")
    config = parse_config(config_file)

    model_title = config["predict"]["model_title"]
    model_review = config["predict"]["model_review"]
    model_classification = config["predict"]["model_classification"]
    data_to_predict = config["predict"]["data_to_predict"]
    predictions_path = Path(config["predict"]["predictions_path"])

    logger.info(f"config: {config['predict']}")

    ##################
    # Load model & data to predict
    ##################
    # Load models
    logger.info(
        f"-------------------Load the trained model-------------------")
    model_title = load_model(model_title)
    model_review = load_model(model_review)
    model_classification = load_model(model_classification)

    # Load data
    logger.info(f"Load the data to predict {data_to_predict}")
    data = pd.read_csv(data_to_predict)
    # Dropping rows without data in content
    data = data.dropna(subset=['content']).reset_index(drop=True)

    # Filtering comments according to keywords
    data['is_transport_related'] = data['content'].str.contains(
        utility.TRANSPORT_KEYWORDS, case=False, na=False)
    data = data[data['is_transport_related'] == True].reset_index(drop=True)
    data = data.drop(columns=['is_transport_related'])

    # Defining noise words
    logger.info("Defining noise words.")
    noise_words = []
    stopwords_corpus = nltk.corpus.stopwords
    eng_stop_words = stopwords_corpus.words('english')
    noise_words.extend(eng_stop_words)

    logger.info("End load model and data to predict")

    ##################
    # NLP for title
    ##################
    logger.info("-------------------NLP for title-------------------")

    exp_name = setup(data=data[['title']],
                     target='title',
                     session_id=42,
                     custom_stopwords=noise_words)
    predictions_title = assign_model(model_title)  # Using pycaret

    predictions_title = predictions_title.add_prefix('Title_')
    predictions_title['Title_Dominant_Topic'] = predictions_title[
        'Title_Dominant_Topic'].replace(' ', '_', regex=True)

    logger.info("End NLP for title")

    ##################
    # NLP for review
    ##################
    logger.info("-------------------NLP for review-------------------")

    exp_name = setup(data=data[['content']],
                     target='content',
                     session_id=42,
                     custom_stopwords=noise_words)
    predictions_review = assign_model(model_review)  # Using pycaret

    predictions_review = predictions_review.add_prefix('Review_')
    predictions_review['Review_Dominant_Topic'] = predictions_review[
        'Review_Dominant_Topic'].replace(' ', '_', regex=True)

    logger.info("End NLP for review")

    ##################
    # Merging data
    ##################
    logger.info(f"-------------------Merging data-------------------")

    data = pd.concat([predictions_title, predictions_review], axis=1)
    logger.info(f"shape: {data.shape}")

    logger.info("End Load and merge data")

    ##################
    # Make prediction and export file
    ##################
    logger.info(f"-------------------Predict and evaluate-------------------")
    predictions = predict_model(model_classification, data)  # Using pycaret
    predictions.to_csv(predictions_path / "data_predicted.csv", index=False)


if __name__ == "__main__":
    predict()
