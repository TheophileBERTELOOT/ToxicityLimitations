import torch
from transformers import pipeline

class Llama:
    def __init__(self) -> None:
        model_id = "meta-llama/Llama-3.2-3B"
        self.model = pipeline(
        "text-generation", 
        model=model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda"
)
        
    def getToxicityScore(self,message):
        result = self.model(message)
        print(result)
        return result




