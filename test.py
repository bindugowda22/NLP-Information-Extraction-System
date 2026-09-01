import streamlit
import spacy
import nltk
import pandas
import networkx
import plotly
import transformers
import torch
import datasets

print("All libraries installed successfully!")

nlp = spacy.load("en_core_web_sm")

print("spaCy model loaded successfully!")