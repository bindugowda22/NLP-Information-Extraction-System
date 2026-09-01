import nltk

resources = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ("chunkers/maxent_ne_chunker", "maxent_ne_chunker"),
    ("corpora/words", "words"),
]

for path, package in resources:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(package)

print("NLTK resources downloaded successfully!")
