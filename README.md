# SentimentMountEtna
![badge1](https://img.shields.io/badge/language-Python-blue.svg)
![badge2](https://img.shields.io/badge/framework-scrapy-green.svg)
![badge3](https://img.shields.io/badge/data-TripAdvisor-lightgray.svg)
> Sentiment analysis towards transport systems in Mount Etna.
## A sentiment analysis approach to understand tourist satisfaction towards transport systems: the case of Mount Etna
This model implements a Latent Dirichlet Allocation method to assign topic modelling, and then implements a Naive Bayes model to perform a sentiment analysis based on title+review 

## How to use  SentimentMountEtna
### 1. Prerequisites
To be able to run this app, you need to have:
* [Python 3.8](https://www.python.org/downloads/)
```
sudo apt install pipenv -y
pipenv shell
pipenv install --skip-lock
```

### 2. Run
Once the configuration is complete, it's time to run this app.
Simply run the following command lines below in a terminal:
```
scrapy runspider ./scripts/get_data.py -o ./data/mount_etna_data.csv -t csv
pipenv run python -m spacy download en_core_web_sm
pipenv run python ./scripts/nlp.py
pipenv run python ./scripts/classification.py
pipenv run python ./scripts/predict.py
```
