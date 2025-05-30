from ToxicityLimitations.Datasets.Subtle import SubtleDataset
import pandas as pd
import math
import time 
import os
from sklearn.metrics import precision_score, recall_score, f1_score

class SubtleExperiments:
    def __init__(self,models,outputPath) -> None:
        self.dataset = SubtleDataset()
        self.models = models
        self.outputPath = outputPath
        
        
    def run(self):
        for modelName in self.models.keys():
            print(modelName)
            i = 0
            if os.path.exists(self.outputPath+modelName+'.csv'):
                df = pd.read_csv(self.outputPath+modelName+'.csv',index_col=0)
                firstRowToBeTreated = len(df)
            else:
                df = pd.DataFrame(columns=['ToxicityBinary','Toxicity','IdentityAttack','Insult','Profanity','Threat','SevereToxicity','Justification','message_id','text'])
                firstRowToBeTreated = 0
            t1 = time.time()
            percentage = 0
            lastPercentage = 0
            for row in self.dataset.data['train']:
                if i > firstRowToBeTreated:
                    model = self.models[modelName]
                    if row['cleaned_text'].replace(" ", "") != '':
                        response = model.getToxicityScore(row['cleaned_text'])
                        response['message_id'] = row['message_id']
                        response['text'] = row['cleaned_text']
                        df.loc[len(df)] = response
                    percentage = math.trunc((i*100)/len(self.dataset.data['train']))
                    if i%10 == 0:
                        print(str(percentage) +'%')
                        print('___________________________________')
                        lastPercentage = percentage
                        df.to_csv(self.outputPath+modelName+'.csv')
                i+=1
            if firstRowToBeTreated < len(self.dataset.data['train'])-2:           
                df.to_csv(self.outputPath+modelName+'.csv')
            timePassed = str((time.time()-t1)/60)
            print(timePassed+' minutes passed')

    def get_true_toxicity_for_text(self,text,data):
        row = data[data['cleaned_text']==text]
        true_toxicity = list(row['hateful_layer'])[0]
        true_implicit = list(row['implicit_layer'])[0]
        true_subtle= list(row['subtlety_layer'])[0]
        if true_toxicity == 'HS':
            true_toxicity = 1 
        else:
            true_toxicity = 0
        if true_implicit == 'Implicit HS':
            true_implicit = 1 
        else:
            true_implicit = 0
        if true_subtle == 'Subtle':
            true_subtle = 1 
        else:
            true_subtle = 0
        return true_toxicity,true_implicit,true_subtle

    def format_predicted_toxicity(self,values):
        print(values.unique())
        values_list = []
        for value in values:
            text = value
            append = False
            if  isinstance(text,str):
                text = text.replace(' ','')
                text = text.lower()
                if text == 'yes':
                    values_list.append(1)
                    append = True
                elif text  == 'no':
                    values_list.append(0)
                    append = True
                elif text == '0':
                    values_list.append(0)
                    append = True
                elif text == '1':
                    values_list.append(0)
                    append = True
                elif text == 'notoxicity':
                    values_list.append(0)
                    append = True
                elif text == 'yes/no-no':
                    values_list.append(0)
                    append = True
                elif 'yes' in text and not 'no' in text:
                    values_list.append(1)
                    append = True
                elif 'no' in text and not 'yes' in text:
                    values_list.append(0)
                    append = True
                elif text =='yes/no':
                    values_list.append(0) 
                    append = True
                elif 'yes(' in text:
                    values_list.append(1)
                    append = True
                elif 'nono' in text:
                    values_list.append(0)
                    append = True
                elif 'noyes' in text:
                    values_list.append(1)
                    append = True
                elif 'notoxicity' in text:
                    values_list.append(0)
                    append = True
                elif '->yes' in text:
                    values_list.append(1)
                    append = True
                elif '->no' in text:
                    values_list.append(0)
                    append = True
                elif 'modified' in text:
                    values_list.append(0)
                    append = True
            else:
                values_list.append(text)
                append = True
            if not append:
                print( text)
        return values_list
    
    def compute_results(self):
        data = pd.DataFrame(self.dataset.data['train'])
        folder = '../results/'
        models = [
            f.split('.')[0] for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f)) and '_results' not in f
        ]
        models.remove('Falcon')
        print(models)
        stats_list = []
        for model in models:
            print(model)
            if not os.path.exists("../results/"+model+'_results.csv'):
                df = pd.read_csv(folder+model+'.csv',index_col=0)
                df = df.dropna(subset=["ToxicityBinary", "text"])
                df_model_results = df[['ToxicityBinary','text']]
                df_model_results['ToxicityBinary'] = self.format_predicted_toxicity(df_model_results['ToxicityBinary'])
                true_toxicity_list = []
                true_subtle_list = []
                true_implicit_list = []
                for idx in df_model_results.index:
                    row = df_model_results.loc[idx]
                    true_toxicity,true_implicit,true_subtle = self.get_true_toxicity_for_text(row['text'],data)
                    true_toxicity_list.append(true_toxicity)
                    true_subtle_list.append(true_subtle)
                    true_implicit_list.append(true_implicit)
                df_model_results['true_toxicity'] = true_toxicity_list
                df_model_results['true_subtle'] = true_subtle_list
                df_model_results['true_implicit'] = true_implicit_list
                df_model_results.to_csv('../results/'+model+'_results.csv')
            else:
                df_model_results = pd.read_csv("../results/"+model+'_results.csv',index_col=0)

            stats_list.append([
                model,
                precision_score(df_model_results['true_toxicity'] ,df_model_results['ToxicityBinary']),
                recall_score(df_model_results['true_toxicity'] ,df_model_results['ToxicityBinary']),
                f1_score(df_model_results['true_toxicity'] ,df_model_results['ToxicityBinary']),
                precision_score(df_model_results['true_subtle'] ,df_model_results['ToxicityBinary']),
                recall_score(df_model_results['true_subtle'] ,df_model_results['ToxicityBinary']),
                f1_score(df_model_results['true_subtle'] ,df_model_results['ToxicityBinary']),
                precision_score(df_model_results['true_implicit'] ,df_model_results['ToxicityBinary']),
                recall_score(df_model_results['true_implicit'] ,df_model_results['ToxicityBinary']),
                f1_score(df_model_results['true_implicit'] ,df_model_results['ToxicityBinary']),
            ])
            df_stats = pd.DataFrame(stats_list,columns=['model','precision_toxicity','recall_toxicity','f1_toxicity',
                                                        'precision_subtle','recall_subtle','f1_subtle',
                                                        'precision_implicit','recall_implicit','f1_implicit'])
            df_stats.to_csv('../results/Subtle_experiment_results.csv')

                

    