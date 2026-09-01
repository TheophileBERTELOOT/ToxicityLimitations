from datasets import load_dataset
import os
#https://zenodo.org/records/48810080
class ConversationDataset:
    def __init__(self) -> None:
        self.data_files = os.listdir('../../data/conversations')


