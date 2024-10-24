from perspective import PerspectiveAPI

class Perspective:
    def __init__(self,perspective_api_key) -> None:
        self.p = PerspectiveAPI(perspective_api_key)
        
    def getToxicityScore(self,message):
        return self.p.score(message,['TOXICITY','SEVERE_TOXICITY','IDENTITY_ATTACK','INSULT','PROFANITY','THREAT'])
    
    