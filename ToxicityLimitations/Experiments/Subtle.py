from ToxicityLimitations.Datasets.Subtle import SubtleDataset
import pandas as pd

class SubtleExperiments:
    def __init__(self,models,outputPath) -> None:
        self.dataset = SubtleDataset()
        self.models = models
        self.outputPath = outputPath
        
        
    def run(self):
        i=0
        for modelName in self.models.keys():
            df = pd.DataFrame(columns=['ToxicityBinary','Toxicity','IdentityAttack','Insult','Profanity','Threat','SevereToxicity','Justification','message_id','text'])
            for row in self.dataset.data['train']:
                print(modelName)
                print(row)
                model = self.models[modelName]
                response = model.getToxicityScore(row['cleaned_text'])
                response['message_id'] = row['message_id']
                response['text'] = row['cleaned_text']
                print(response)
                df.loc[len(df)] = response
                i+=1
                if i == 2:
                    break
            df.to_csv(self.outputPath+modelName+'.csv')
    