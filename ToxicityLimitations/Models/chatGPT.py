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
        return completion.choices[0].message
        

        
        


