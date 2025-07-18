from datasets import load_dataset
#https://zenodo.org/records/48810080
class ConversationDataset:
    def __init__(self) -> None:
        self.data = load_dataset("jigsaw_toxicity_pred")


