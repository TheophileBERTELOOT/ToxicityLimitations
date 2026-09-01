from ToxicityLimitations.Datasets.Conversation import ConversationDataset
import pandas as pd
import math
import time 
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from ToxicityLimitations.Models.RobertaToxicGen import RoBERTa_ToxiGen
from ToxicityLimitations.Models.HateBertToxicGen import Hatebert_toxigen
# from ToxicityLimitations.Models.Perspective import Perspective
from ToxicityLimitations.Models.Llama import Llama
from ToxicityLimitations.Models.LlamaGuard import LlamaGuard
# from ToxicityLimitations.Models.Gemini import Gemini
# from ToxicityLimitations.Models.claude import Claude
from ToxicityLimitations.Models.Falcon import Falcon
from ToxicityLimitations.Models.Mistral import Mistral
from ToxicityLimitations.Models.Aya import Aya
from ToxicityLimitations.Models.DolphinLlama import DolphinLlama
from ToxicityLimitations.Models.Gemma2 import Gemma2
from ToxicityLimitations.Models.Granite3Guardian import Granite3Guardian
from ToxicityLimitations.Models.Hermes3 import Hermes3
from ToxicityLimitations.Models.Llama3Chat import Llama3ChatQA
from ToxicityLimitations.Models.MistralOrca import MistralOpenOrca
from ToxicityLimitations.Models.NeuralChat import NeuralChat
from ToxicityLimitations.Models.OpenHermes import Openhermes
# from ToxicityLimitations.Models.Phi35 import Phi35
from ToxicityLimitations.Models.Qwen25 import Qwen25
from ToxicityLimitations.Models.Contexts import get_conversation_context
from tqdm import tqdm

    

               
class ConversationExperiments:
    def __init__(self,models,outputPath) -> None:
        self.dataset = ConversationDataset()
        self.models = models
        self.outputPath = outputPath

    def select_model(self,modelName):
        context = Contexts.Toxicity.value
        models_path = "/lustre06/project/6010878/thber64/ToxicityLimitationsProject/ToxicityLimitations/media/theophileberteloot/ad5f5ae6-6fd8-46ef-8659-a07d195c6d70/PhD/ToxicityLimitations/Models/"
        if modelName == 'Aya':
            return Aya(context,isOffline=True,model_path=models_path) 
        if modelName == 'DolphinLlama':
            return DolphinLlama(context,isOffline=True,model_path=models_path)
        if modelName == 'Gemma2':
            return Gemma2(context,isOffline=True,model_path=models_path)
        if modelName == 'Granite3Guardian':
            return Granite3Guardian(context,isOffline=True,model_path=models_path)
        if modelName == 'Qwen25':
            return Qwen25(context,isOffline=True,model_path=models_path)
        if modelName == 'Hermes3':
            return Hermes3(context,isOffline=True,model_path=models_path)
        if modelName == 'Llama3ChatQA':
            return Llama3ChatQA(context,isOffline=True,model_path=models_path)
        if modelName == 'MistralOpenOrca':
            return MistralOpenOrca(context,isOffline=True,model_path=models_path)
        if modelName == 'NeuralChat':
            return NeuralChat(context,isOffline=True,model_path=models_path)
        if modelName == 'OpenHermes':
            return OpenHermes(context,isOffline=True,model_path=models_path)
        if modelName == 'Roberta':
            return RoBERTa_ToxiGen(isOffline=True,model_path=models_path)
        if modelName == 'Hatebert':
            return Hatebert_toxigen(isOffline=True,model_path=models_path)
        if modelName == 'Llama':
            return Llama(context,isOffline=True,model_path=models_path)
        if modelName == 'LlamaGuard':
            return LlamaGuar(context,isOffline=True,model_path=models_path)
        if modelName == 'Falcon':
            return Falcon(context,isOffline=True,model_path=models_path)
        if modelName == 'Mistral':
            return Mistral(context,isOffline=True,model_path=models_path)
        
        
    def create_prompts(self,conversation_df,message_id):
        conversational_context = '\n'.join(list(conversation_df.iloc[:message_id]['meta_text']))

        message = conversation_df.iloc[message_id]['meta_text']
        prompt = get_conversation_context(conversational_context,message)
        print(prompt,flush=True)
        return prompt
        
    def run(self):
        for modelName in self.models.keys():
            print(modelName)
            i = 0
            if os.path.exists(self.outputPath+modelName+'.csv'):
                df = pd.read_csv(self.outputPath+modelName+'.csv',index_col=0)
                last_row = df.iloc[-1]
                firstRowToBeTreated =int(last_row['conversation_id'])
            else:
                df = pd.DataFrame(columns=['toxicity_binary','toxicity','identity_attack','insult','profanity','threat','severe_toxicity','justification','message_id','text','conversation_id'])
                firstRowToBeTreated = 0
            t1 = time.time()
            percentage = 0
            lastPercentage = 0
            model = self.select_model(self.models[modelName])
            data = self.dataset.data_files.select(range(firstRowToBeTreated+1,len(self.dataset.data_files)))

            for c in tqdm(range(len(data)), desc="Processing batches"):
                conversation_df = pd.read_csv(c)
                for m_id in range(len(conversation_df)):
                    message = conversation_df.iloc[m_id]
                    prompt = self.create_prompts(conversation_df,m_id)
                    res = model.getToxicityScores([prompt])

                
                    if 'error' not in res:
                        res[0]['message_id'] = message['id']
                        res[0]['text'] = message['meta_text'] 
                        res[0]['conversation_id'] = c
                    else:
                        # Si la génération ou le parsing échoue
                        res[0]['message_id'] = message['id']
                        res[0]['text'] = message['meta_text'] 
                        res[0]['conversation_id'] = c
                    df.loc[len(df)] = res                    
                if i%(10000*self.batch_size) == 0:
                    print(str(percentage) +'%')
                    print('___________________________________')
                    lastPercentage = percentage
                    df.to_csv(self.outputPath+modelName+'.csv')
                i+=self.batch_size
            if firstRowToBeTreated < len(self.dataset.data)-2:           
                df.to_csv(self.outputPath+modelName+'.csv')
            timePassed = str((time.time()-t1)/60)
            print(timePassed+' minutes passed')

    

                

    