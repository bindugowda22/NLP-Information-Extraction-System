from modules.ner import named_entity_recognition

text = """
Microsoft announced a partnership with OpenAI in Seattle on Monday.
"""

entity_df, doc = named_entity_recognition(text)

print(entity_df)