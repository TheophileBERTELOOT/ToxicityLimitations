from transformers import pipeline

class RoBERTa_ToxiGen:
    def __init__(self) -> None:
        self.model = pipeline("text-classification", model="tomh/toxigen_roberta",device='cuda')
        
    def getToxicityScore(self,message):
        result = self.model(message)
        return self.getFormatedResponse(result[0])
    
    def getFormatedResponse(self,response):
        formatedResponse = {'toxicity binary':0,'toxicity Score':0}
        if response['label'] == 'LABEL_0':
            formatedResponse['toxicity binary'] = 0
            formatedResponse['toxicity'] = 1 - response['score']
        else:
            formatedResponse['toxicity binary'] = 1
            formatedResponse['toxicity'] = response['score']
        return formatedResponse
            

