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
                    return {'ToxicityBinary':-1,'Toxicity':-1,'IdentityAttack':-1,'Insult':-1,'Profanity':-1,'Threat':-1,'SevereToxicity':-1,'Justification':''}
        
    def getFormatedResponse(self,response):
        formatedResponse = {'ToxicityBinary':0,'Toxicity':0,'IdentityAttack':0,'Insult':0,'Profanity':0,'Threat':0,'SevereToxicity':0,'Justification':''}
        formatedResponse['ToxicityBinary'] = float(response['TOXICITY']) > 0.5
        formatedResponse['IdentityAttack'] = response['IDENTITY_ATTACK']
        formatedResponse['Insult'] = response['INSULT']
        formatedResponse['Profanity'] = response['PROFANITY']
        formatedResponse['Threat'] = response['THREAT']
        formatedResponse['SevereToxicity'] = response['SEVERE_TOXICITY']
        formatedResponse['Toxicity'] = response['TOXICITY']
        return formatedResponse
    
    