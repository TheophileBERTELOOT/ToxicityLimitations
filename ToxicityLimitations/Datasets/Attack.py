from datasets import load_dataset

class AttackDataset:
    def __init__(self) -> None:
        self.data = load_dataset("jigsaw_toxicity_pred")


