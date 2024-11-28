from ToxicityLimitations.Datasets.Subtle import SubtleDataset
import pandas as pd
import math
import time 

class SubtleExperiments:
    def __init__(self,models,outputPath) -> None:
        self.dataset = SubtleDataset()
        self.models = models
        self.outputPath = outputPath
        
        
    def run(self):
        for modelName in self.models.keys():
            df = pd.DataFrame(columns=['ToxicityBinary','Toxicity','IdentityAttack','Insult','Profanity','Threat','SevereToxicity','Justification','message_id','text'])
            i = 0
            t1 = time.time()
            percentage = 0
            lastPercentage = 0
            for row in self.dataset.data['train']:
                model = self.models[modelName]
                response = model.getToxicityScore(row['cleaned_text'])
                response['message_id'] = row['message_id']
                response['text'] = row['cleaned_text']
                print(response)
                df.loc[len(df)] = response
                i+=1
                percentage = math.trunc((i*100)/len(self.dataset.data['train']))
                if percentage > lastPercentage:
                    print(str(percentage) +'%')
                    print('___________________________________')
                    lastPercentage = percentage
                    df.to_csv(self.outputPath+modelName+'.csv')              
            df.to_csv(self.outputPath+modelName+'.csv')
            timePassed = str((time.time()-t1)/60)
            print(timePassed+' minutes passed')
    