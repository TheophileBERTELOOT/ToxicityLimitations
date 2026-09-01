from perspective import PerspectiveAPI
import time

class Perspective:
    def __init__(self,perspective_api_key) -> None:
        self.p = PerspectiveAPI(api_key=perspective_api_key)
        
    def getToxicityScore(self,message):
        if message.replace(" ", "") != '':
            try:
                return self.getFormatedResponse(self.p.score(message,['TOXICITY','SEVERE_TOXICITY','IDENTITY_ATTACK','INSULT','PROFANITY','THREAT']))
            except:
                time.sleep(60)
                print('message : '+message)
                try:
                    return self.getFormatedResponse(self.p.score(message,['TOXICITY','SEVERE_TOXICITY','IDENTITY_ATTACK','INSULT','PROFANITY','THREAT']))
                except:
                    return {'toxicity_binary':0,'toxicity':0,'identity_attack':0,'insult':0,'profanity':0,'threat':0,'severe_toxicity':0,'justification':''}
        
    def getFormatedResponse(self,response):
        formatedResponse = {'toxicity_binary':0,'toxicity':0,'identity_attack':0,'insult':0,'profanity':0,'threat':0,'severe_toxicity':0,'justification':''}
        formatedResponse['toxicity_binary'] = float(response['TOXICITY']) > 0.5
        formatedResponse['identity_attack'] = response['IDENTITY_ATTACK']
        formatedResponse['insult'] = response['INSULT']
        formatedResponse['profanity'] = response['PROFANITY']
        formatedResponse['threat'] = response['THREAT']
        formatedResponse['severe_toxicity'] = response['SEVERE_TOXICITY']
        formatedResponse['toxicity'] = response['TOXICITY']
        return formatedResponse
    
    