import anthropic

class Claude:
    def __init__(self,api_key,context) -> None:  
        self.client = anthropic.Anthropic(

    api_key=api_key,
) 
        self.context = context


    def getToxicityScore(self,message:str):
        response = self.client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": self.context},
        {"role": "user", "content": message}
    ]
)
        return self.getFormatedResponse(response.content[0].text)
        
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

        
        


