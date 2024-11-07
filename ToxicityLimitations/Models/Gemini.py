import pathlib
import textwrap

import google.generativeai as genai
from google.generativeai import caching



class Gemini:
    def __init__(self,api_key,context) -> None:  

        genai.configure(api_key=api_key)
        cache = caching.CachedContent.create(
            model='gemini-1.5-pro',
            display_name='toxicity limitation', 
            system_instruction=(
                context
            )
        )
        self.model = genai.GenerativeModel.from_cached_content(cached_content=cache)

    def getToxicityScore(self,message:str):
        response = self.model.generate_content(message)
        print(response.prompt_feedback)
        return response.text


