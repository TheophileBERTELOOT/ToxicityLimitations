from ollama import chat
from ollama import ChatResponse

class Qwen25:
    def __init__(self,context) -> None:
        self.context = context
        
    def getToxicityScore(self,message):
        response: ChatResponse = chat(model='qwen2.5', messages=[
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
        response = response.split('\n')
        for field in response:
            field = field.split(':')
            key = field[0]
            key = key.replace(' ','')
            if key in formatedResponse.keys():
                formatedResponse[key] = field[1]
        return formatedResponse




