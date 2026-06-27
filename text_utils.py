"""
Shared resume-text cleaning.

Both the training script and the prediction API import this file, so a
resume is cleaned exactly the same way at train time and at inference
time. Inconsistent cleaning between training and serving is one of the
most common (and hardest to debug) causes of a model that scores well
in testing but performs poorly in production.
"""
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data once, quietly, if not already present.
for package in ("stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{package}")
    except LookupError:
        nltk.download(package, quiet=True)

CUSTOM_STOPWORDS = {
    "engineering", "engineer", "student", "professional", "highly", "skills",
    "experience", "work", "history", "details", "academic", "profile", "specialize",
    "specializing", "experienced", "projects", "project", "related", "responsibility",
    "tutor", "teaching", "tutions", "tutor", "graphic", "designer", "design", "wordpress",
    "technical", "like", "class", "advanced", "peer", "subject", "idea", "world", "home",
    "food", "school", "taught", "grade", "science"
}
STOP_WORDS = set(stopwords.words("english")).union(CUSTOM_STOPWORDS)
LEMMATIZER = WordNetLemmatizer()

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")


def clean_resume(text: str) -> str:
    """Lowercase, strip noise/URLs/emails, drop stopwords, lemmatize."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)

    words = text.split()
    words = [
        LEMMATIZER.lemmatize(word)
        for word in words
        if word not in STOP_WORDS and len(word) > 1
    ]
    return " ".join(words)
