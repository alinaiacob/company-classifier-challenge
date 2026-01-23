import re

def normalize_text(text):
    text = str(text).lower()  # lowercase
    text = re.sub(r'\s+', ' ', text)  # remove multiple whitespaces
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuations
    text = text.strip()
    return text