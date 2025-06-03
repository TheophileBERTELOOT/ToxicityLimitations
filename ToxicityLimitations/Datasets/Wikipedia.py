import pandas as pd

class WikipediaDataset:
    def __init__(self) -> None:
        self.data = pd.read_csv('../data/Wikipedia/train.csv',index_col=0)


