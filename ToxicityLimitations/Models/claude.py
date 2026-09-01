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
        formatedResponse = {'toxicity_binary':0,'toxicity':0,'identity_attack':0,'insult':0,'profanity':0,'threat':0,'severe_toxicity':0,'justification':''}
        response = response.split('\n')
        for field in response:
            field = field.split(':')
            key = field[0]
            key = key.replace(' ','')
            if key in formatedResponse.keys():
                formatedResponse[key] = field[1]
        return formatedResponse

        
        


