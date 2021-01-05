import numpy as np

from news_classifier.constants import LABEL_COL, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN, TITLE_COL
from news_classifier.preprocessing import clean_headline, tokenize_headlines


class Vocabulary:

    def __init__(
        self,
        token_to_index: dict=None,
        tokens: list=None,
        unk_token: bool=True,
        sos_token: bool=True,
        eos_token: bool=True
        ):
        """
        """
        self._token_to_index = {} if token_to_index is None else token_to_index
        self._tokens = [] if tokens is None else tokens
        self.unk_token = unk_token
        self.sos_token = sos_token
        self.eos_token = eos_token

        if unk_token:
            self.add_token(UNK_TOKEN)
        if sos_token:
            self.add_token(SOS_TOKEN)
        if eos_token:
            self.add_token(EOS_TOKEN)

    def add_token(self, token):
        if token in self._token_to_index:
            return
        index = len(self._tokens)
        self._token_to_index[token] = index
        self._tokens.append(token)

    @classmethod
    def from_serializable(cls, dictionary):
        return cls(**dictionary)

    def to_serializable(self):
        return {
            'token_to_index': self._token_to_index,
            'tokens': self._tokens,
            'unk_token': self.unk_token,
            'sos_token': self.sos_token,
            'eos_token': self.eos_token
        }

    def lookup_token(self, token):
        if token in self._token_to_index:
            return self._token_to_index[token]
        if self.unk_token:
            return self._token_to_index[UNK_TOKEN]
        raise KeyError(f'Token {token} does not belong to this vocab and uknown tokens are turned off')

    def lookup_index(self, index):
        if index >= len(self._tokens):
            raise KeyError(f'Index {index} does not belong to this vocab')
        return self._tokens[index]

    def __len__(self):
        return len(self._tokens)



class Vectorizer:

    def __init__(self, headlines_vocab, labels_vocab):
        self.headlines_vocab = headlines_vocab
        self.labels_vocab = labels_vocab

    def vectorize_headline(self, headline):
        cleaned_headline = clean_headline(headline)
        tokens = tokenize_headlines(cleaned_headline)

        one_hot = np.zeros(len(self.headlines_vocab), dtype=np.float32)
        for token in tokens:
            index = self.headlines_vocab.lookup_token(token)
            one_hot[index]  = 1
        return one_hot

    def vectorize_category(self, category):
        return self.labels_vocab.lookup_token(category)

    @classmethod
    def from_dataframe(cls, df):
        headlines_vocab = Vocabulary()
        labels_vocab = Vocabulary(sos_token=False, eos_token=False, unk_token=False)
        for _, row in df.iterrows():
            headline, label = row[TITLE_COL], row[LABEL_COL]
            cleaned_headline = clean_headline(headline)
            tokens = tokenize_headlines(cleaned_headline)
            for token in tokens:
                headlines_vocab.add_token(token)
            labels_vocab.add_token(label)
        return cls(headlines_vocab, labels_vocab)

    @classmethod
    def from_serializable(cls, contents):
        headlines_vocab = Vocabulary.from_serializable(contents['headlines_vocab'])
        labels_vocab = Vocabulary.from_serializable(contents['labels_vocab'])
        return cls(headlines_vocab, labels_vocab)

    def to_serializable(self):
        return {
            'headlines_vocab': self.headlines_vocab.to_serializable(),
            'labels_vocab': self.labels_vocab.to_serializable()
               }
