from datasets import load_dataset

class WikipediaDataset:
    def __init__(self) -> None:
        self.data = load_dataset("csv", data_files="Data/wikipedia.csv",split='train')



