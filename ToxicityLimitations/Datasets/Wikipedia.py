from datasets import load_dataset

class WikipediaDataset:
    def __init__(self) -> None:
        self.data = load_dataset("BenjaminOcampo/ISHate")


