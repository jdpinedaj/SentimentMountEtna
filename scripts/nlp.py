# -*- coding: utf-8 -*-
"""
This script is used to extract the dimensions in the titles and the reviews of the data.

Usage:
    python3 ./scripts/nlp.py

"""
##################
# Importing libraries
##################
import logging
from pathlib import Path
from utility import parse_config, set_logger
import utility
import click
from pycaret.nlp import *
import pandas as pd
import numpy as np
import spacy

spacy.load("en_core_web_sm")

# NLTK
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet, stopwords

#ignore log(0) and divide by 0 warning
np.seterr(divide='ignore')
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


@click.command()
@click.argument("config_file", type=str, default="scripts/config.yml")
def nlp(config_file):
    """
    NLP function that extracts the dimensions in the titles and the reviews of the data.

    Args:
        config_file [str]: path to config file

    Returns:
        None
    """

    ##################
    # configure logger
    ##################
    logger = set_logger("./log/nlp.log")

    ##################
    # Load config from config file
    ##################
    logger.info(f"Load config from {config_file}")
    config = parse_config(config_file)

    raw_data_file = config["nlp"]["raw_data_file"]
    model_title_path = config["nlp"]["model_title_path"]
    model_review_path = config["nlp"]["model_review_path"]
    dimensions_path = Path(config["nlp"]["dimensions_path"])

    logger.info(f"config: {config['nlp']}")

    ##################
    # Data transformation and Feature engineering
    ##################
    logger.info(
        "-------------------Start data transformation and feature engineering-------------------"
    )
    df = pd.read_csv(raw_data_file, low_memory=False)
    data = df[['title', 'content', 'rating']]

    # Dropping rows without data in content
    data = data.dropna(subset=['content']).reset_index(drop=True)

    # Filtering comments according to keywords
    logger.info("Filtering comments according to keywords.")
    data['is_transport_related'] = data['content'].str.contains(
        utility.TRANSPORT_KEYWORDS, case=False, na=False)
    data = data[data['is_transport_related'] == True].reset_index(drop=True)
    data = data.drop(columns=['is_transport_related'])
    logger.info(f"shape: {data.shape}")

    # Creating sentiment_rating feature based on rating
    logger.info("Creating sentiment_rating feature based on rating.")
    data['sentiment_rating'] = np.where(data['rating'] > 3, 1, 0)

    # changing bad and good just for visualization in the recall metric
    data['sentiment_rating'] = data['sentiment_rating'].replace(
        [0, 1], ['negative', 'positive'])
    data['sentiment_rating'] = data['sentiment_rating'].replace(
        ['negative', 'positive'], [1, 0])  # NEGATIVE IS 1!!!!

    # Defining noise words
    logger.info("Defining noise words.")
    noise_words = []
    stopwords_corpus = nltk.corpus.stopwords
    eng_stop_words = stopwords_corpus.words('english')
    noise_words.extend(eng_stop_words)

    logger.info("End data transformation and feature engineering")

    ##################
    # NLP for title
    ##################
    logger.info("-------------------NLP for title-------------------")

    # Setup Pycaret
    logger.info(f"Setup Pycaret")

    exp_name = setup(data=data[['title', 'sentiment_rating']],
                     target='title',
                     session_id=42,
                     custom_stopwords=noise_words)

    # Training model
    logger.info(f"Training model")

    logger.info(f"Tuning model")
    tuned_lda_title = tune_model(model='lda',
                                 multi_core=True,
                                 supervised_target='sentiment_rating',
                                 custom_grid=[2, 3, 4, 5, 6],
                                 optimize='AUC',
                                 verbose=False)

    lda_title_tuned = create_model(
        'lda', multi_core=True,
        num_topics=tuned_lda_title.num_topics)  # Latent Dirichlet Allocation
    lda_title_data_tuned = assign_model(lda_title_tuned)
    lda_title_data_tuned = lda_title_data_tuned.add_prefix('Title_')
    lda_title_data_tuned['Title_Dominant_Topic'] = lda_title_data_tuned[
        'Title_Dominant_Topic'].replace(' ', '_', regex=True)

    logger.info("End NLP for title")

    ##################
    # NLP for review
    ##################
    logger.info("-------------------NLP for review-------------------")

    # Setup Pycaret
    logger.info(f"Setup Pycaret")

    exp_name = setup(data=data[['content', 'sentiment_rating']],
                     target='content',
                     session_id=42,
                     custom_stopwords=noise_words)

    # Training model
    logger.info(f"Training model")

    logger.info(f"Tuning model")
    tuned_lda_review = tune_model(model='lda',
                                  multi_core=True,
                                  supervised_target='sentiment_rating',
                                  custom_grid=[2, 3, 4, 5, 6],
                                  optimize='AUC',
                                  verbose=False)

    lda_review_tuned = create_model(
        'lda', multi_core=True,
        num_topics=tuned_lda_review.num_topics)  # Latent Dirichlet Allocation

    lda_review_data_tuned = assign_model(lda_review_tuned)
    lda_review_data_tuned = lda_review_data_tuned.add_prefix('Review_')
    lda_review_data_tuned['Review_Dominant_Topic'] = lda_review_data_tuned[
        'Review_Dominant_Topic'].replace(' ', '_', regex=True)

    logger.info("End NLP for review")

    ##################
    # Export
    ##################
    logger.info("-------------------Export-------------------")
    logger.info(f"write data to {dimensions_path}")

    lda_title_data_tuned.to_csv(dimensions_path / "lda_title_data_tuned.csv",
                                index=False)
    lda_review_data_tuned.to_csv(dimensions_path / "lda_review_data_tuned.csv",
                                 index=False)
    logger.info(f"lda_title_data_tuned shape: {lda_title_data_tuned.shape}")
    logger.info(f"lda_review_data_tuned shape: {lda_review_data_tuned.shape}")
    logger.info("\n")

    logger.info("End Export")

    ##################
    # Saving models
    ##################

    logger.info(f"-------------------Saving models-------------------")
    save_model(lda_title_tuned, model_title_path)
    save_model(lda_review_tuned, model_review_path)
    logger.info(f"End saving models")


if __name__ == "__main__":
    nlp()
