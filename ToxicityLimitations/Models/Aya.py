from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import pipeline
import os
from huggingface_hub import snapshot_download
import outlines
from ToxicityLimitations.Models.Contexts import ToxicityResultsFormat
import re
import json
from pydantic import BaseModel,ValidationError

class Aya:
    def __init__(self,context,isOffline=False,model_path='') -> None:
        self.context = context
        self.model_name = 'CohereLabs/aya-expanse-8b'
        self.model_path=model_path
        if not isOffline:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", torch_dtype=torch.float16)
            self.model.eval()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path+self.model_name,
 local_files_only=True,device_map='auto',low_cpu_mem_usage=True,use_fast=True)
            self.model     = AutoModelForCausalLM.from_pretrained(model_path+self.model_name, local_files_only=True,device_map='auto',low_cpu_mem_usage=True,torch_dtype='auto')
            self.model.eval()

            self.gen = outlines.from_transformers(self.model,self.tokenizer)
        torch.set_grad_enabled(False)
 
    def getToxicityScore(self,message):
        return json.loads(self.gen(self.context + message,ToxicityResultsFormat, max_new_tokens=256,do_sample=False,use_cache=True,temperature=None, top_p=None))



    def safe_json_parse_with_pydantic(self,text: str):
        """Extrait et valide la sortie JSON selon la classe ToxicityResultsFormat."""
        print('_______________________')
        print(text)
        try :
            text = '{'+text.split("{", 1)[1]
            if '"' not in text[-2:]:   # vérifie les 2 derniers caractères
                text += '"'
            if "}" not in text:
                text += "}"
        except:
            print('error')
            return {"error": "no_json_found", "raw": text}
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            print('error')
            return {"error": "no_json_found", "raw": text}

        json_text = match.group(0)
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as e:
            print('error')
            return {"error": "json_decode_error", "details": str(e), "raw": text}

        try:
            parsed = ToxicityResultsFormat(**data)
            return parsed.model_dump()  
        except ValidationError as e:
            print('error')
            return {"error": "validation_failed", "details": str(e), "raw": text}
        
    def getToxicityScores(self,messages):
        prompts = [self.context + msg for msg in messages]
        enc = self.tokenizer(prompts,return_tensors='pt',padding=True, truncation=True)
        enc = {k: v.to(self.model.device, non_blocking=True) for k, v in enc.items()} 
        with torch.inference_mode():
            outputs = self.model.generate(
                **enc,
                max_new_tokens=156,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        gen_only = outputs[:, enc["input_ids"].shape[1]:]
        texts = self.tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        parsed = [self.safe_json_parse_with_pydantic(t) for t in texts]
        return parsed

    def downloadModel(self):
        snapshot_download(repo_id='CohereLabs/aya-expanse-8b', repo_type="model", local_dir="/media/theophileberteloot/ad5f5ae6-6fd8-46ef-8659-a07d195c6d70/PhD/ToxicityLimitations/Models/"+self.model_name+"/model")

    def getFormatedResponse(self,response):
        print(response)
        formatedResponse = {'toxicity_binary':0,'toxicity':0,'identity_attack':0,'insult':0,'profanity':0,'threat':0,'severe_toxicity':0,'justification':''}
        response = response.split('\n')
        for field in response:
            field = field.split(':')
            key = field[0]
            key = key.replace(' ','')
            if key in formatedResponse.keys():
                formatedResponse[key] = field[1]
        return formatedResponse




