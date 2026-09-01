from Datasets.Toxigen import ToxigenDataset
import pandas as pd
import math
import time 
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
import numpy as np
# from Models.RobertaToxicGen import RoBERTa_ToxiGen
# from Models.HateBertToxicGen import Hatebert_toxigen
# # from Models.Perspective import Perspective
# from Models.Llama import Llama
# from Models.LlamaGuard import LlamaGuard
# # from Models.Gemini import Gemini
# # from Models.claude import Claude
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
# # from Models.Phi35 import Phi35
# from Models.Qwen25 import Qwen25
from Models.chatGPT import chatGPTVanilla
from Models.Contexts import Contexts
from tqdm import tqdm

class ToxigenExperiments:
    def __init__(self,models,outputPath) -> None:
        self.dataset = ToxigenDataset()
        self.models = models
        self.outputPath = outputPath
        self.batch_size = 32
 
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
            return Openhermes(context,isOffline=True,model_path=models_path)
        if modelName == 'Roberta':
            return RoBERTa_ToxiGen(isOffline=True,model_path=models_path)
        if modelName == 'Hatebert':
            return Hatebert_toxigen(isOffline=True,model_path=models_path)
        if modelName == 'Llama':
            return Llama(context,isOffline=True,model_path=models_path)
        if modelName == 'LlamaGuard':
            return LlamaGuard(context,isOffline=True,model_path=models_path)
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
            percentage = 0
            lastPercentage = 0
            print(firstRowToBeTreated)
            model = self.select_model(self.models[modelName],open_api_key,org,project)
            data = self.dataset.data.select(range(firstRowToBeTreated+1,len(self.dataset.data)))
            n_batches = math.ceil(len(data) / self.batch_size)
            for b in tqdm(range(n_batches), desc="Processing batches"):

                batch = data[b * self.batch_size : (b + 1) * self.batch_size]
                prompts = batch['text']
                if modelName == 'ChatGPT':
                    outputs = []
                    for prompt in prompts:
                        output = model.getToxicityScore(prompt)
                        outputs.append(output)
                else:
                    outputs = model.getToxicityScores(prompts)
                
                for text, res in zip(batch['text'], outputs):
                    if 'error' not in res:

                        res['text'] = text 
                    else:
                        # Si la g  n  ration ou le parsing   choue
                        res['text'] = text
                        res['error'] = res.get('error', 'unknown')
                    df.loc[len(df)] = res
                if i%(10*self.batch_size) == 0:
                    print(str(percentage) +'%')
                    print('___________________________________')
                    lastPercentage = percentage
                    df.to_csv(self.outputPath+modelName+'.csv')
                i+=self.batch_size
            if firstRowToBeTreated < len(self.dataset.data):           
                df.to_csv(self.outputPath+modelName+'.csv')
            timePassed = str((time.time()-t1)/60)
            print(timePassed+' minutes passed')

    def clean_binary_column(self,series):
        return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({
            "true": "1",
            "false": "0",
            "yes": "1",
            "no": "0",
            "nan": np.nan,
            "none": np.nan,
            "": np.nan
        })
        .pipe(pd.to_numeric, errors="coerce")
    )

    def compute_results(self):
        labels_df = pd.read_csv('Data/toxigen.csv')
        true_labels = labels_df['label']
        true_labels = true_labels.eq('hate')
        folder = 'results/Toxigen/'
        models = [
            f.split('.')[0] for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f)) and '_results' not in f
        ]

        results = []
        for model in models:
            print(model)
            model_df = pd.read_csv(folder+model+'.csv')
            aligned_df = model_df[["text", "toxicity_binary"]].merge(
    labels_df[["text", "label"]],
    left_on="text",
    right_on="text",
    how="inner"
)
            aligned_df = aligned_df.dropna(subset=["toxicity_binary"])
            aligned_df = aligned_df.dropna(subset=["label"])
            aligned_df['label'] = aligned_df['label'] == 'hate'
            true_labels = aligned_df['label'].astype(int)
            predicted_labels = self.clean_binary_column(aligned_df['toxicity_binary'])


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
        results_df.to_csv('results/Toxigen/toxigen_results.csv')


        


    