from ToxicityLimitations.Datasets.Subtle import SubtleDataset


class SubtleExperiments:
    def __init__(self,models,outputPath) -> None:
        self.dataset = SubtleDataset()
        self.models = models
        self.outputPath = outputPath
        
    def run(self):
        i=0
        for row in self.dataset.data['train']:
            for modelName in self.models.keys():
                print(modelName)
                print(row)
                model = self.models[modelName]
                response = model.getToxicityScore(row['cleaned_text'])
                print(response)
                # f = open(self.outputPath+modelName+'.csv', "a")
                # f.write()
            i+=1
            if i == 2:
                break
    