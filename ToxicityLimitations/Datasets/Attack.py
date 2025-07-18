from datasets import load_dataset
#https://zenodo.org/records/6615386 
class AttackDataset:
    def __init__(self) -> None:
        self.data = load_dataset("jigsaw_toxicity_pred")


