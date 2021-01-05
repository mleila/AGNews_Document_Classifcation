import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from news_classifier.constants import DATASET, TITLE_COL, TRAIN, VALID, TEST, LABEL_COL
from news_classifier.language import Vectorizer

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, vectorizer, split=TRAIN):
        self.df = df
        self.vectorizer = vectorizer

        self.set_split(split)

    def set_split(self, split=TRAIN):
        self._target_df = self.df.query(f'{DATASET} == "{split}"')

    @classmethod
    def from_dataframe(cls, df):
        vectorizer = Vectorizer.from_dataframe(df)
        return cls(df, vectorizer)

    def __getitem__(self, index):
        row = self._target_df.iloc[index]
        headline, label = row[TITLE_COL], row[LABEL_COL]
        headline_vector = self.vectorizer.vectorize_headline(headline)
        label_vector = self.vectorizer.vectorize_category(label)
        return {'x': headline_vector, 'y': label_vector}

    def __len__(self):
        return len(self._target_df)


def assign_rows_to_split(
    df: pd.DataFrame,
    train_ratio: float=0.7,
    valid_ratio: float=0.15,
    test_ratio: float=0.15
    ):
    """
    Assign each row to either a training, validation, or testing datasets

    Args:
     - df: pandas dataframe with two columns, news headline and label
     - train_ratio: ratio of training data
     - valid_ratio: ratio of validation data
     - test_ratio: ratio of testing data

    returns:
     - dataframe with an added column (dataset) with either train, test, or valid
    """
    assert train_ratio + valid_ratio + test_ratio == 1, 'splitting ratios must add to one'

    train_rows, non_train_rows = train_test_split(
        df,
        train_size=train_ratio,
        shuffle=True,
        stratify=df[LABEL_COL]
        )

    valid_rows, test_rows = train_test_split(
        non_train_rows,
        train_size=valid_ratio/(valid_ratio+test_ratio),
        stratify=non_train_rows[LABEL_COL]
        )

    train_rows[DATASET] = TRAIN
    valid_rows[DATASET] = VALID
    test_rows[DATASET] = TEST
    return pd.concat([train_rows, valid_rows, test_rows], axis=0)


def generate_batches(
    dataset,
    batch_size,
    shuffle=True,
    drop_last=True,
    device='cpu'
    ):
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
    for data_dict in data_loader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = tensor.to(device)
        yield out_data_dict
