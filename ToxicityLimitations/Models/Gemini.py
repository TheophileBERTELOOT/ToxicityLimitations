import pathlib
import textwrap

import google.generativeai as genai
from google.generativeai import caching



class Gemini:
    def __init__(self,api_key,context) -> None:  

        genai.configure(api_key=api_key)
        self.context = context
        # cache = caching.CachedContent.create(
        #     model='gemini-1.5-flash',
        #     display_name='toxicity limitation', 
        #     system_instruction=(
        #         context
        #     )
        # )
        self.model = genai.GenerativeModel("gemini-1.5-pro")

    def getToxicityScore(self,message:str):
        response = self.model.generate_content(self.context+'\n'+message)
        return self.getFormatedResponse(response.text)
    
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


