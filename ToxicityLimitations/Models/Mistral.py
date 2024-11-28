from ollama import chat
from ollama import ChatResponse

class Mistral:
    def __init__(self,api_key,context) -> None:  
        self.context = context


    def getToxicityScore(self,message:str):
        response: ChatResponse = chat(model='llama3.2', messages=[
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

        
        


