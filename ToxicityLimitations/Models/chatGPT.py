from openai import OpenAI


from Models.Contexts import ToxicityResultsFormat
import json
import re
import json
from pydantic import ValidationError


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
        return self.safe_json_parse_with_pydantic_2(completion.choices[0].message.content)


    def safe_json_parse_with_pydantic_2(self, text: str):
        """
        Extrait et valide la sortie JSON selon la classe ToxicityResultsFormat.
        Robuste aux réponses du type:
        ```json
        {...}
        ```
        ou
        Here is the JSON: {...}
        """

       

        raw_text = text

    # Enlever les blocs markdown éventuels
        text = text.strip()
        text = text.replace("```json", "")
        text = text.replace("```JSON", "")
        text = text.replace("```", "")
        text = text.strip()

    # Chercher tous les objets JSON potentiels
        candidates = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
        if not candidates:
        # fallback : si le modèle a oublié l'accolade finale
            start = text.rfind("{")
            if start == -1:
                return {
                    "error": "no_json_found",
                "raw": raw_text
            }

            candidate = text[start:].strip()

            if not candidate.endswith("}"):
                candidate += "}"

            candidates = [candidate]

        last_error = None

        for json_text in reversed(candidates):
            json_text = json_text.strip()
            try:
                data = json.loads(json_text)
            except json.JSONDecodeError as e:
                last_error = e
                continue
            try:
                parsed = ToxicityResultsFormat(**data)
                return parsed.model_dump()

            except ValidationError as e:
                return {
                "error": "validation_failed",
                "details": str(e),
                "json_text": json_text,
                "raw": raw_text
            }

        return {
        "error": "json_decode_error",
        "details": str(last_error),
        "raw": raw_text
    } 
        
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
        
        


