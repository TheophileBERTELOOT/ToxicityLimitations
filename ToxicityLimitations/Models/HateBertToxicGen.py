from transformers import pipeline

class Hatebert_toxigen:
    def __init__(self) -> None:
        self.model = pipeline("text-classification", model="tomh/toxigen_hatebert", tokenizer="bert-base-uncased",device='cuda')
        
    def getToxicityScore(self,message):
        result = self.model(message)
        return self.getFormatedResponse(result[0])
    
    def getFormatedResponse(self,response):
        formatedResponse = {'ToxicityBinary':0,'Toxicity':0,'IdentityAttack':0,'Insult':0,'Profanity':0,'Threat':0,'SevereToxicity':0,'Justification':''}
        if response['label'] == 'LABEL_0':
            formatedResponse['ToxicityBinary'] = 0
            formatedResponse['Toxicity'] = 1 - response['score']
        else:
            formatedResponse['ToxicityBinary'] = 1
            formatedResponse['Toxicity'] = response['score']
        return formatedResponse

