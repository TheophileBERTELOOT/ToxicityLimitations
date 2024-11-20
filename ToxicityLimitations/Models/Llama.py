import torch
from transformers import pipeline

class Llama:
    def __init__(self,context) -> None:
        model_id = "meta-llama/Llama-3.2-3B"
        self.model = pipeline(
        "text-generation", 
        model=model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda"
)
        self.context = context
        
    def getToxicityScore(self,message):
        result = self.model(self.context+'\n'+message)
        return self.getFormatedResponse(result)
    
    def getFormatedResponse(self,response):
        formatedResponse = {'Toxicity Binary':0,'Toxicity':0,'Identity attack':0,'Insult':0,'Profanity':0,'Threat':0,'Severe Toxicity':0,'Justification':''}
        response = response.split('\n')
        for field in response:
            field = field.split(':')
            if field[0] in formatedResponse.keys():
                formatedResponse[field[0]] = field[1]
        return formatedResponse




