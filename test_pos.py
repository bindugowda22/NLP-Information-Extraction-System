from modules.pos_tagger import pos_tagging

text = """
Microsoft announced a partnership with OpenAI in Seattle on Monday.
"""

spacy_df, nltk_df = pos_tagging(text)

print("SpaCy POS")
print(spacy_df)

print("\nNLTK POS")
print(nltk_df)