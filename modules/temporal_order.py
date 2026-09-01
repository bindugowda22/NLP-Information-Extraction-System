import pandas as pd
import re
from datetime import datetime
import dateparser
from .common import get_nlp

def temporal_ordering(text):
    doc = get_nlp()(text)
    rows = []

    for sent in doc.sents:
        sentence = sent.text.strip()
        # First use spaCy's DATE/TIME entities.
        time_texts = [ent.text for ent in sent.ents if ent.label_ in {"DATE", "TIME"}]

        # Also recognize common explicit relative/ordinal temporal phrases.
        for phrase in re.findall(
            r"\b(?:yesterday|today|tomorrow|last week|next week|last month|next month)\b",
            sentence,
            flags=re.I,
        ):
            if phrase not in time_texts:
                time_texts.append(phrase)

        for time_text in time_texts:
            parsed = dateparser.parse(time_text, settings={"PREFER_DATES_FROM": "past"})
            rows.append({
                "Time Expression": time_text,
                "Parsed Date": parsed.strftime("%Y-%m-%d %H:%M:%S") if parsed else "",
                "Event/Sentence": sentence,
            })

    if not rows:
        return pd.DataFrame(columns=["Time Expression", "Parsed Date", "Event/Sentence"])

    return pd.DataFrame(rows).sort_values(
        by=["Parsed Date", "Time Expression"], na_position="last"
    ).reset_index(drop=True)
