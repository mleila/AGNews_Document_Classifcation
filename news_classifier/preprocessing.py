import string



def clean_headline(headline: str):
    cleaned = headline.lower()
    words = cleaned.split(" ")
    words = [word for word in words if word not in string.punctuation]
    cleaned = ' '.join(words)
    return cleaned


def tokenize_headlines(headline:str):
    tokens = headline.split(" ")
    return tokens
