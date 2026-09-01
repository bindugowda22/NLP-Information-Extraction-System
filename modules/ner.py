import pandas as pd
from .common import get_nlp

def named_entity_recognition(text):
    doc = get_nlp()(text)
    rows = [
        {
            "Entity": ent.text,
            "Label": ent.label_,
            "Start": ent.start_char,
            "End": ent.end_char,
        }
        for ent in doc.ents
    ]
    return pd.DataFrame(rows), doc
