from transformers import pipeline

class RoBERTa_ToxiGen:
    def __init__(self) -> None:
        self.model = pipeline("text-classification", model="tomh/toxigen_roberta")
        
    def getToxicityScore(self,message):
        result = self.model(message)
        print(result)
        return result

