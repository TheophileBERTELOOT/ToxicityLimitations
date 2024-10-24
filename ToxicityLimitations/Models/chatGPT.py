from openai import OpenAI


class chatGPTVanilla:
    def __init__(self,open_api_key,org,project) -> None:  
        self.client =  OpenAI(organization= org,
                            project= project,
                            api_key=open_api_key)


    def getToxicityScore(self,message:str):
        completion = self.client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": message}
        ]
        )
        print(completion.choices[0].message)


