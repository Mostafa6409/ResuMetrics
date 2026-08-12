import spacy

nlp = spacy.load("en_core_web_sm")

def preprocess_Text(text):
  doc = nlp(text.lower())
  clean_tokens = [
        token.lemma_ for token in doc 
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]
  return " ".join(clean_tokens)
  