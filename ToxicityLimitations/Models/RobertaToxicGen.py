from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

class RoBERTa_ToxiGen:
    def __init__(self,isOffline=False) -> None:
        if not isOffline:
            self.model = pipeline("text-classification", model="tomh/toxigen_roberta",tokenizer="bert-base-uncased",device='cuda')
        else:
            if os.path.exists('../Models/toxigen_roberta'):
                self.model = pipeline(
                "text-classification",
                model="../Models/toxigen_roberta",
                tokenizer="../Models/bert-base-uncased",
                device=0
                )
            else:
                print('model is not here dl it first')


        
    def downloadModel(self):
        model = AutoModelForSequenceClassification.from_pretrained("tomh/toxigen_roberta")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model.save_pretrained("../Models/toxigen_roberta")
        tokenizer.save_pretrained("../Models/bert-base-uncased")


    def getToxicityScore(self,message):
        result = self.model(message)
        return self.getFormatedResponse(result[0])
    
    def getFormatedResponse(self,response):
        print(response)
        formatedResponse = {'ToxicityBinary':0,'Toxicity':0,'IdentityAttack':0,'Insult':0,'Profanity':0,'Threat':0,'SevereToxicity':0,'Justification':''}
        if response['label'] == 'LABEL_0':
            formatedResponse['ToxicityBinary'] = 0
            formatedResponse['Toxicity'] = 1 - response['score']
        else:
            formatedResponse['ToxicityBinary'] = 1
            formatedResponse['Toxicity'] = response['score']
        return formatedResponse
            

