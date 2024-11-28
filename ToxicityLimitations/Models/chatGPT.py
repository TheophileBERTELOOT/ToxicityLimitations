from openai import OpenAI


class chatGPTVanilla:
    def __init__(self,open_api_key,org,project,context) -> None:  
        self.client =  OpenAI(organization= org,
                            project= project,
                            api_key=open_api_key)
        self.context = context


    def getToxicityScore(self,message:str):
        completion = self.client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content" : self.context},
            {"role": "user", "content": message}
        ]
        )
        return self.getFormatedResponse(completion.choices[0].message.content)
        
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
        
        


