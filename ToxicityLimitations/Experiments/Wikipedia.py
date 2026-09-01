from Datasets.Wikipedia import WikipediaDataset
import pandas as pd
import math
import time 
import os

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
# from Models.RobertaToxicGen import RoBERTa_ToxiGen
# from Models.HateBertToxicGen import Hatebert_toxigen
# from Models.Perspective import Perspective
# from Models.Llama import Llama
# from Models.LlamaGuard import LlamaGuard
# from Models.Gemini import Gemini
# from Models.claude import Claude
# from Models.Falcon import Falcon
# from Models.Mistral import Mistral
# from Models.Aya import Aya
# from Models.DolphinLlama import DolphinLlama
# from Models.Gemma2 import Gemma2
# from Models.Granite3Guardian import Granite3Guardian
# from Models.Hermes3 import Hermes3
# from Models.Llama3Chat import Llama3ChatQA
# from Models.MistralOrca import MistralOpenOrca
# from Models.NeuralChat import NeuralChat
# from Models.OpenHermes import Openhermes
# from Models.Phi35 import Phi35
# from Models.Qwen25 import Qwen25
from Models.Contexts import Contexts
from tqdm import tqdm
from Models.chatGPT import chatGPTVanilla

class WikipediaExperiments:
    def __init__(self,models,outputPath) -> None:
        self.dataset = WikipediaDataset()
        self.models = models
        self.outputPath = outputPath
        self.batch_size = 16
        
    def select_model(self,modelName,open_api_key='',org='',project=''):
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
        if modelName == 'ChatGPT':
            return chatGPTVanilla(open_api_key,org,project,context)
               
    def run(self,open_api_key='',org='',project=''):
        
        for modelName in self.models.keys():
            print(modelName)
            i = 0
            if os.path.exists(self.outputPath+modelName+'.csv'):
                df = pd.read_csv(self.outputPath+modelName+'.csv',index_col=0)
                firstRowToBeTreated = len(df)
            else:
                df = pd.DataFrame(columns=['toxicity_binary','toxicity','identity_attack','insult','profanity','threat','severe_toxicity','justification','message_id','text'])
                firstRowToBeTreated = 0
            t1 = time.time()
            data = self.dataset.data.select(range(firstRowToBeTreated+1,len(self.dataset.data)))
            n_batches = math.ceil(len(data) / self.batch_size)
            model = self.select_model(self.models[modelName],open_api_key,org,project)
            for b in tqdm(range(n_batches), desc="Processing batches"):
                batch = data[b * self.batch_size : (b + 1) * self.batch_size]
                prompts = batch['comment_text']
                if modelName == 'ChatGPT':
                    outputs = []
                    for prompt in prompts:
                        output = model.getToxicityScore(prompt)
                        outputs.append(output)
                else:
                    outputs = model.getToxicityScores(prompts)
                for msg_id,text, res in zip(batch['id'],batch['comment_text'], outputs):
                    print('----------------------------')
                    if 'error' not in res:
                        res['id'] = msg_id
                        res['text'] = text 
                    else:
                        print('on devrait pas etre la')
                        # Si la génération ou le parsing échoue
                        res['text'] = text
                        res['id'] = msg_id
                        res['error'] = res.get('error', 'unknown')
                    df.loc[len(df)] = res                    
                if i%(10*self.batch_size) == 0:
                    df.to_csv(self.outputPath+modelName+'.csv')
                i+=self.batch_size
            if firstRowToBeTreated < len(data):           
                df.to_csv(self.outputPath+modelName+'.csv')
            timePassed = str((time.time()-t1)/60)
            print(timePassed+' minutes passed')

    def compute_results(self):

        
        labels_df = pd.read_csv('Data/wikipedia.csv')
        print(labels_df.columns)
        folder = 'results/Wikipedia/'
        models = [
            f.split('.')[0] for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f)) and '_results' not in f
        ]

        results = []
        for model in models:
            print(model)
            model_df = pd.read_csv(folder+model+'.csv')
            aligned_df = model_df[["text", "toxicity_binary"]].merge(
    labels_df[["comment_text", "toxic"]],
    left_on="text",
    right_on="comment_text",
    how="inner"
)
            aligned_df = aligned_df.dropna(subset=["toxicity_binary"])
            aligned_df = aligned_df.dropna(subset=["toxic"])
            true_labels = (aligned_df['toxic'] > 0.5).astype(int)
            predicted_labels = aligned_df['toxicity_binary'].astype(int)

            precision = precision_score(true_labels, predicted_labels)
            recall = recall_score(true_labels, predicted_labels)
            f1 = f1_score(true_labels, predicted_labels)
            accuracy = accuracy_score(true_labels, predicted_labels)

            results.append({'model':model,
                            'precision':precision,
                            'recall':recall,
                            'f1':f1,
                            'accuracy':accuracy,})
            

        results_df = pd.DataFrame(results)
        print(results_df)
        results_df.to_csv('results/Wikipedia/wikipedia_results.csv')
    
