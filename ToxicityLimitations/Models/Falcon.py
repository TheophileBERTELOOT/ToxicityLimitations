from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Falcon:
    def __init__(self,context) -> None:
        self.context = context
        self.model_name = 'tiiuae/falcon-7b'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", torch_dtype=torch.float16)
        self.model.eval()
        
    def getToxicityScore(self,message):
        inputs = self.tokenizer(self.context+message, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            do_sample=True,
        )
        return self.getFormatedResponse(outputs[0]) 
    
    def getFormatedResponse(self,response):
        formatedResponse = {'ToxicityBinary':0,'Toxicity':0,'IdentityAttack':0,'Insult':0,'Profanity':0,'Threat':0,'SevereToxicity':0,'Justification':''}
        response = response.split('\n')
        for field in response:
            field = field.split(':')
            key = field[0]
            key = key.replace(' ','')
            if key in formatedResponse.keys():
                formatedResponse[key] = field[1]
        return formatedResponse




