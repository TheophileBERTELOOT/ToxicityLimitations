from ollama import chat
from ollama import ChatResponse

class Granite3Guardian:
    def __init__(self,context) -> None:
        self.context = context
        
    def getToxicityScore(self,message):
        response: ChatResponse = chat(model='granite3-guardian:8b', messages=[
            {
                'role': 'system',
                'content': self.context,
            },{
                'role': 'user',
                'content' : message
            }
            ])
        return self.getFormatedResponse(response['message']['content']) 
    
    def getFormatedResponse(self,response):
        formatedResponse = {'ToxicityBinary':0,'Toxicity':0,'IdentityAttack':0,'Insult':0,'Profanity':0,'Threat':0,'SevereToxicity':0,'Justification':''}
        if response != 'No':
            formatedResponse['ToxicityBinary'] = 1
        return formatedResponse




