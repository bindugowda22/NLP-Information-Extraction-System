import pandas as pd
from .common import get_nlp

def relation_extraction(text):
    doc = get_nlp()(text)
    rows = []

    # Lightweight rule-based relation extraction:
    # for each sentence, connect nearby named entities through a meaningful verb.
    for sent in doc.sents:
        entities = list(sent.ents)
        verbs = [t for t in sent if t.pos_ in {"VERB", "AUX"}]
        if len(entities) >= 2:
            relation = verbs[0].lemma_ if verbs else "related_to"
            for i in range(len(entities) - 1):
                rows.append({
                    "Subject": entities[i].text,
                    "Relation": relation,
                    "Object": entities[i + 1].text,
                })

    return pd.DataFrame(rows, columns=["Subject", "Relation", "Object"])
