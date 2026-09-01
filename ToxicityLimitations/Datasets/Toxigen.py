from datasets import load_dataset

class ToxigenDataset:
    def __init__(self) -> None:
        self.data = load_dataset("csv", data_files="Data/toxigen.csv",split='train')


