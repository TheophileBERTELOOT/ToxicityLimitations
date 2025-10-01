from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import pipeline
import os
from huggingface_hub import snapshot_download
import outlines
from ToxicityLimitations.Models.Contexts import ToxicityResultsFormat
import json
class Mistral:
    def __init__(self,context,isOffline=False,model_path='') -> None:
        self.context = context
        self.model_name = 'mistralai/Ministral-8B-Instruct-2410'
        self.model_path = model_path
        if not isOffline:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", torch_dtype=torch.float16)
            self.model.eval()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path+self.model_name,local_files_only=True,device_map='auto',low_cpu_mem_usage=True,use_fast=True)
            self.model     = AutoModelForCausalLM.from_pretrained(model_path+self.model_name, local_files_only=True,device_map='auto',low_cpu_mem_usage=True,torch_dtype='auto')
            self.model.eval()
            self.gen = outlines.from_transformers(self.model,self.tokenizer)
        
    def getToxicityScore(self,message):
        return json.loads(self.gen(self.context + message,ToxicityResultsFormat, max_new_tokens=256,do_sample=False,use_cache=True,temperature=None, top_p=None))
    def downloadModel(self):
        snapshot_download(repo_id=self.model_name, repo_type="model", local_dir="/media/theophileberteloot/ad5f5ae6-6fd8-46ef-8659-a07d195c6d70/PhD/ToxicityLimitations/Models/"+self.model_name)

    
        
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

        
        


