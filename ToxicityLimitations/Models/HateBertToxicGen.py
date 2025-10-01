from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os 


class Hatebert_toxigen:
    def __init__(self,isOffline=False,model_path='') -> None:
        self.model_path=model_path
        self.model_name='tomh/toxigen_hatebert'
        if not isOffline:
            self.model = pipeline("text-classification", model=self.model_name,tokenizer="bert-base-uncased",device='cuda')
        else:
            if os.path.exists(model_path+self.model_name):
                self.model = pipeline(
                "text-classification",
                model=model_path+self.model_name,
                tokenizer=model_path+"bert-base-uncased",
                device='auto'
                )
            else:
                print('model is not here dl it first')

    def downloadModel(self):
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model.save_pretrained(self.model_path+self.model_name)
        tokenizer.save_pretrained(self.model_path+"bert-base-uncased")
        
    def getToxicityScore(self,message):
        result = self.model(message)
        return self.getFormatedResponse(result[0])
    
    def getFormatedResponse(self,response):
        formatedResponse = {'toxicity_binary':0,'toxicity':0,'identity_attack':0,'insult':0,'profanity':0,'threat':0,'severe_toxicity':0,'justification':'',}
        if response['label'] == 'LABEL_0':
            formatedResponse['ToxicityBinary'] = 0
            formatedResponse['Toxicity'] = 1 - response['score']
        else:
            formatedResponse['ToxicityBinary'] = 1
            formatedResponse['Toxicity'] = response['score']
        return formatedResponse

