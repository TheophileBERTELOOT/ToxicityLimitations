from transformers import pipeline

class Hatebert_toxigen:
    def __init__(self) -> None:
        self.model = pipeline("text-classification", model="tomh/toxigen_hatebert", tokenizer="bert-base-uncased",device='cuda')
        
    def getToxicityScore(self,message):
        result = self.model(message)
        return result

