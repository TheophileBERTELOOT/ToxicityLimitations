from perspective import PerspectiveAPI

class Perspective:
    def __init__(self,perspective_api_key) -> None:
        self.p = PerspectiveAPI(perspective_api_key)
        
    def getToxicityScore(self,message):
        return self.getFormatedResponse(self.p.score(message,['TOXICITY','SEVERE_TOXICITY','IDENTITY_ATTACK','INSULT','PROFANITY','THREAT']))
    
    def getFormatedResponse(self,response):
        formatedResponse = {'Toxicity Binary':0,'Toxicity':0,'Identity attack':0,'Insult':0,'Profanity':0,'Threat':0,'Severe Toxicity':0,'Justification':''}
        formatedResponse['Toxicity Binary'] = response['TOXICITY'] > 0.5
        formatedResponse['Identity attack'] = response['IDENTITY_ATTACK']
        formatedResponse['Insult'] = response['INSULT']
        formatedResponse['Profanity'] = response['PROFANITY']
        formatedResponse['Threat'] = response['THREAT']
        formatedResponse['Severe Toxicity'] = response['SEVERE_TOXICITY']
        return formatedResponse
    
    