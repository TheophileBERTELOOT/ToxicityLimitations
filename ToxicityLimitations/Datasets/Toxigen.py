from datasets import load_dataset

class ToxigenDataset:
    def __init__(self) -> None:
        self.data = load_dataset("toxigen/toxigen-data", "annotated")


