from datasets import load_dataset
#https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification
class BiaisDataset:
    def __init__(self) -> None:
        self.data = load_dataset("csv", data_files="../Data/biaisDataset.csv",split='train')


