from ToxicityLimitations.Datasets.Subtle import SubtleDataset
import pandas as pd
import math
import time 
import os

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
    