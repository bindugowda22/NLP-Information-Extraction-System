import pandas as pd
import nltk
from nltk import word_tokenize, pos_tag
from .common import get_nlp

def _ensure_nltk():
    for package, resource in [
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
        ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
    ]:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, quiet=True)

def pos_tagging(text):
    doc = get_nlp()(text)
    spacy_df = pd.DataFrame([
        {
            "Token": token.text,
            "POS": token.pos_,
            "Tag": token.tag_,
            "Dependency": token.dep_,
        }
        for token in doc
    ])

    _ensure_nltk()
    try:
        nltk_tags = pos_tag(word_tokenize(text))
    except LookupError:
        nltk_tags = []
    nltk_df = pd.DataFrame(nltk_tags, columns=["Token", "POS"])
    return spacy_df, nltk_df
