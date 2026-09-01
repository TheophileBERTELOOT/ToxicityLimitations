from Datasets.Attack import AttackDataset
import pandas as pd
import math
import time 
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
# from ToxicityLimitations.Models.RobertaToxicGen import RoBERTa_ToxiGen
# from ToxicityLimitations.Models.HateBertToxicGen import Hatebert_toxigen
# # from ToxicityLimitations.Models.Perspective import Perspective
# from ToxicityLimitations.Models.Llama import Llama
# from ToxicityLimitations.Models.LlamaGuard import LlamaGuard
# # from ToxicityLimitations.Models.Gemini import Gemini
# # from ToxicityLimitations.Models.claude import Claude
# from ToxicityLimitations.Models.Falcon import Falcon
# from ToxicityLimitations.Models.Mistral import Mistral
# from ToxicityLimitations.Models.Aya import Aya
# from ToxicityLimitations.Models.DolphinLlama import DolphinLlama
# from ToxicityLimitations.Models.Gemma2 import Gemma2
# from ToxicityLimitations.Models.Granite3Guardian import Granite3Guardian
# from ToxicityLimitations.Models.Hermes3 import Hermes3
# from ToxicityLimitations.Models.Llama3Chat import Llama3ChatQA
# from ToxicityLimitations.Models.MistralOrca import MistralOpenOrca
# from ToxicityLimitations.Models.NeuralChat import NeuralChat
# from ToxicityLimitations.Models.OpenHermes import Openhermes
# # from ToxicityLimitations.Models.Phi35 import Phi35
# from ToxicityLimitations.Models.Qwen25 import Qwen25
from Models.Contexts import Contexts
from tqdm import tqdm
class AttackExperiments:
    def __init__(self,models,outputPath) -> None:
        # self.dataset = AttackDataset()
        self.models = models
        self.outputPath = outputPath
        self.batch_size = 64
        

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
                             
        
    def run(self):
        for modelName in self.models.keys():
            print(modelName)
            i = 0
            if os.path.exists(self.outputPath+modelName+'.csv'):
                df = pd.read_csv(self.outputPath+modelName+'.csv',index_col=0)
                firstRowToBeTreated = len(df)
            else:
                df = pd.DataFrame(columns=['original_toxicity_binary',
                                           'original_toxicity',
                                           'original_identity_attack',
                                           'original_insult',
                                           'original_profanity',
                                           'original_threat',
                                           'original_severe_toxicity',
                                           'original_justification',
                                           'unique_src_instance_identifier',
                                           'original_text',
                                           'perturbed_toxicity_binary',
                                           'perturbed_toxicity',
                                           'perturbed_identity_attack',
                                           'perturbed_insult',
                                           'perturbed_profanity',
                                           'perturbed_threat',
                                           'perturbed_severe_toxicity',
                                           'perturbed_justification',
                                           'perturbed_text'])
                firstRowToBeTreated = 0
            t1 = time.time()
            percentage = 0
            lastPercentage = 0
            for row in tqdm(self.dataset.data):
                if i > firstRowToBeTreated:
                    model = self.models[modelName]
                    if row['original_text'].replace(" ", "") != '':
                        response = model.getToxicityScore(row['original_text'])
                        response['unique_src_instance_identifier'] = row['unique_src_instance_identifier']
                        response['original_toxicity_binary'] = response.pop('toxicity_binary')
                        response['original_toxicity'] = response.pop('toxicity')
                        response['original_identity_attack'] = response.pop('identity_attack')
                        response['original_insult'] = response.pop('insult')
                        response['original_profanity'] = response.pop('profanity')
                        response['original_threat'] = response.pop('threat')
                        response['original_severe_toxicity'] = response.pop('severe_toxicity')
                        response['original_justification'] = response.pop('justification')

                        response['original_text'] = row['original_text']
                        response['perturbed_text'] = row['perturbed_text']
                        perturbed_response = model.getToxicityScore(row['perturbed_text'])
                        response['perturbed_toxicity_binary'] = perturbed_response['toxicity_binary']
                        response['perturbed_toxicity'] = perturbed_response['toxicity']
                        response['perturbed_identity_attack'] = perturbed_response['identity_attack']
                        response['perturbed_insult'] = perturbed_response['insult']
                        response['perturbed_profanity'] = perturbed_response['profanity']
                        response['perturbed_threat'] = perturbed_response['threat']
                        response['perturbed_severe_toxicity'] = perturbed_response['severe_toxicity']
                        response['perturbed_justification'] = perturbed_response['justification']

                        df.loc[len(df)] = response
                    percentage = math.trunc((i*100)/len(self.dataset.data))
                    if i%10 == 0:
                        print(str(percentage) +'%')
                        print('___________________________________')
                        lastPercentage = percentage
                        df.to_csv(self.outputPath+modelName+'.csv')
                i+=1
            if firstRowToBeTreated < len(self.dataset.data):           
                df.to_csv(self.outputPath+modelName+'.csv')
            timePassed = str((time.time()-t1)/60)
            print(timePassed+' minutes passed')

    def compute_results(self):

        
        labels_df = pd.read_csv('Data/attaques.csv')
        print(labels_df.columns)
        folder = 'results/attack/'
        models = [
            f.split('.')[0] for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f)) and '_results' not in f
        ]

        results = []
        for model in models:
            print(model)
            model_df = pd.read_csv(folder+model+'.csv')
            aligned_df = model_df[['original_toxicity_binary',"perturbed_text", "perturbed_toxicity_binary"]].merge(
    labels_df[["perturbed_text", "ground_truth"]],
    left_on="perturbed_text",
    right_on="perturbed_text",
    how="inner"
)
            aligned_df = aligned_df[aligned_df["perturbed_toxicity_binary"] != -1].copy()
            aligned_df = aligned_df[aligned_df["original_toxicity_binary"] != -1].copy()
            aligned_df = aligned_df.dropna(subset=["perturbed_toxicity_binary"])
            aligned_df = aligned_df.dropna(subset=["original_toxicity_binary"])
            
            true_labels = aligned_df['ground_truth'].astype(int)
            predicted_labels_perturbed = aligned_df['perturbed_toxicity_binary'].astype(int)
            predicted_labels_original = aligned_df['original_toxicity_binary'].astype(int)
            precision_original = precision_score(true_labels, predicted_labels_original)
            recall_original = recall_score(true_labels, predicted_labels_original)
            f1_original = f1_score(true_labels, predicted_labels_original)
            accuracy_original = accuracy_score(true_labels, predicted_labels_original)

            precision_perturbed = precision_score(true_labels, predicted_labels_perturbed)
            recall_perturbed = recall_score(true_labels, predicted_labels_perturbed)
            f1_perturbed = f1_score(true_labels, predicted_labels_perturbed)
            accuracy_perturbed = accuracy_score(true_labels, predicted_labels_perturbed)

            results.append({'model':model,
                            'precision_original':precision_original,
                            'recall_original':recall_original,
                            'f1_original':f1_original,
                            'accuracy_original':accuracy_original,
                            'precision_perturbed':precision_perturbed,
                            'recall_perturbed':recall_perturbed,
                            'f1_perturbed':f1_perturbed,
                            'accuracy_perturbed':accuracy_perturbed,})
            

        results_df = pd.DataFrame(results)
        print(results_df)
        results_df.to_csv('results/attack/attack_results.csv')

    
                

    