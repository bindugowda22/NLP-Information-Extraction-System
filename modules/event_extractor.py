import pandas as pd
from .common import get_nlp

def event_extraction(text):
    doc = get_nlp()(text)
    rows = []

    for sent in doc.sents:
        for token in sent:
            if token.pos_ == "VERB":
                subjects = [c.text for c in token.children if c.dep_ in {"nsubj", "nsubjpass"}]
                objects = [c.text for c in token.children if c.dep_ in {"dobj", "obj", "attr", "pobj"}]
                rows.append({
                    "Event": token.lemma_,
                    "Trigger": token.text,
                    "Subject": ", ".join(subjects),
                    "Object": ", ".join(objects),
                    "Sentence": sent.text.strip(),
                })

    return pd.DataFrame(
        rows, columns=["Event", "Trigger", "Subject", "Object", "Sentence"]
    )
