from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import pipeline
import os


class Aya:
    def __init__(self,context,isOffline=False) -> None:
        self.context = context
        self.model_name = 'CohereLabs/aya-expanse-8b'
        if not isOffline:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", torch_dtype=torch.float16)
            self.model.eval()
        else:
            if os.path.exists("/media/theophileberteloot/ad5f5ae6-6fd8-46ef-8659-a07d195c6d70/PhD/ToxicityLimitations/Models/"+self.model_name+"/model"):
                    self.model = pipeline(
                    "text-classification",
                    model="/media/theophileberteloot/ad5f5ae6-6fd8-46ef-8659-a07d195c6d70/PhD/ToxicityLimitations/Models/"+self.model_name+"/model",
                    tokenizer="/media/theophileberteloot/ad5f5ae6-6fd8-46ef-8659-a07d195c6d70/PhD/ToxicityLimitations/Models/"+self.model_name+"/tokenizer",
                    device=0
                    )
            else:
                print('model is not here dl it first')
        
    def getToxicityScore(self,message):
        inputs = self.tokenizer(self.context+message, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens = 1024
        )
        gen_text=self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(gen_text)
        return self.getFormatedResponse(gen_text)

    def downloadModel(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model.save_pretrained("/media/theophileberteloot/ad5f5ae6-6fd8-46ef-8659-a07d195c6d70/PhD/ToxicityLimitations/Models/"+self.model_name+"/model")
        tokenizer.save_pretrained("./media/theophileberteloot/ad5f5ae6-6fd8-46ef-8659-a07d195c6d70/PhD/ToxicityLimitations/Models/"+self.model_name+"/tokenizer") 
    
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




