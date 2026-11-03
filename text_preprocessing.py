
import re

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


stemmer = PorterStemmer()

# Preserve negation words such as not and no.
STOP_WORDS = set(ENGLISH_STOP_WORDS) - {
    "no", "not", "nor"
}


def transform_text(text):

    if not isinstance(text, str):
        return ""

    text = text.lower()

    tokens = re.findall(r"[a-z0-9]+", text)

    processed_tokens = [
        stemmer.stem(word)
        for word in tokens
        if word not in STOP_WORDS
    ]

    return " ".join(processed_tokens)
