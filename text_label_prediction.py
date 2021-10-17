# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 14:00:43 2021

@author: Pushpendu
"""

# Import Libraries
import re
import unicodedata
# import contractions
from catboost import CatBoostClassifier

import warnings

warnings.filterwarnings("ignore")

# Loading the model from disk
model = CatBoostClassifier()
model.load_model(r'C:\Users\91991\Documents\anaconda\Assignments\Mr. Cooper\model_deployment\text_classification\model\cb_model.cbm')


# Perform Cleaning process
# Removing URLs
def remove_urls(text):
    text = re.sub(r'http\S+', ' ', text, flags=re.MULTILINE)
    text = re.sub(r'https\S+', ' ', text, flags=re.MULTILINE)
    text = re.sub(r'www\S+', ' ', text, flags=re.MULTILINE)
    return (text)


# Removing bracketed words
def remove_bracket_words(text):
    text = re.sub("[\(\[].*?[\)\]]", " ", text)
    return text


# Normalizing contracted words
def remove_contraction_word(text):
    text = " ".join([contractions.fix(i) for i in text.split() if len(i) > 0 and "@" not in i])
    return text


# Removing ascii characters and symbols words
def remove_ascii_character(text):
    text = re.sub('[^.,:a-zA-Z0-9 \n\.]', ' ', text)
    text = re.sub(' +', ' ', text)
    return text.strip()


# Normalizing accented words
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


# Cleaning Pipeline
def cleaning_pipeline(text):
    cleaned_text = remove_urls(text)
    cleaned_text = remove_bracket_words(cleaned_text)
#     cleaned_text = remove_contraction_word(cleaned_text)
    cleaned_text = remove_ascii_character(cleaned_text)
    cleaned_text = remove_accented_chars(text)

    return cleaned_text


# Model Prediction
def prediction(text):
    pred = model.predict([text])
    return pred[0]


# Main pipeline
def statement_prediction_main_pipeline(text):
    text = cleaning_pipeline(text)
    pred_res = prediction(text)
    confidence_score = max(model.predict_proba([text]))*100
    return pred_res, round(confidence_score,2)



if __name__ == "__main__":
    text = "help me"
    print("PipeLine output:", statement_prediction_main_pipeline(text))