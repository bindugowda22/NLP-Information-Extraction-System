import spacy

_NLP = None

def get_nlp():
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError as exc:
            raise RuntimeError(
                "spaCy model en_core_web_sm is not installed. "
                "Run: python -m spacy download en_core_web_sm"
            ) from exc
    return _NLP
