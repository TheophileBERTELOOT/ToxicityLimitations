import click
import configparser
from ToxicityLimitations.Models.Models import Models
from ToxicityLimitations.Datasets.Datasets import Datasets
from ToxicityLimitations.Experiments.Experiments import Experiments
from ToxicityLimitations.Models.Contexts import Contexts

# from ToxicityLimitations.Models.chatGPT import chatGPTVanilla
# from ToxicityLimitations.Models.RobertaToxicGen import RoBERTa_ToxiGen
from ToxicityLimitations.Models.HateBertToxicGen import Hatebert_toxigen
# from ToxicityLimitations.Models.Perspective import Perspective
# from ToxicityLimitations.Models.Llama import Llama
# from ToxicityLimitations.Models.LlamaGuard import LlamaGuard
# from ToxicityLimitations.Models.Gemini import Gemini
# from ToxicityLimitations.Models.claude import Claude
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
# from ToxicityLimitations.Models.Phi35 import Phi35
# from ToxicityLimitations.Models.Qwen25 import Qwen25


from ToxicityLimitations.Datasets.Subtle import SubtleDataset

from ToxicityLimitations.Experiments.Subtle import SubtleExperiments
from ToxicityLimitations.Experiments.Toxigen import ToxigenExperiments
from ToxicityLimitations.Experiments.Attack import AttackExperiments
from ToxicityLimitations.Experiments.Conversation import ConversationExperiments
from ToxicityLimitations.Experiments.Wikipedia import WikipediaExperiments
from ToxicityLimitations.Experiments.Biais import BiaisExperiments




def cli():
    context = Contexts.Toxicity.value
    models_path = "/media/theophileberteloot/ad5f5ae6-6fd8-46ef-8659-a07d195c6d70/PhD/ToxicityLimitations/Models/"

    models = {
            # Models.aya.value : Aya(context,isOffline=True),
            # Models.dolphin_llama3.value : DolphinLlama(context),
            # Models.Gemma2.value  : Gemma2(context),
            # Models.granite3_guardian.value : Granite3Guardian(context),
            # Models.Qwen25.value : Qwen25(context),
            # Models.hermes3.value : Hermes3(context),
            # Models.llama3_chatqa.value : Llama3ChatQA(context),
            # Models.mistral_openorca.value: MistralOpenOrca(context),
            # Models.neural_chat.value : NeuralChat(context),
            # Models.openhermes.value : Openhermes(context),
            # Models.phi35.value : Phi35(context),
            # Models.ChatGPT.value : chatGPTVanilla(openai_api_key,org_key,project_key,context),
            # Models.RoBERTa_ToxiGen.value : RoBERTa_ToxiGen(isOffline=True,model_path=models_path),
            Models.Hatebert_toxigen.value : Hatebert_toxigen(isOffline=True,model_path=models_path),
            # Models.Perspective.value : Perspective(perspective_api_key),
            # Models.Gemini.value : Gemini(gemini_api_key,context),
            # Models.Claude.value : Claude(claude_api_key,context),
            # Models.Llama.value : Llama(context),
            # Models.LlamaGuard.value: LlamaGuard(context)
            # Models.Falcon.value : Falcon(context),
            # Models.Mistral.value : Mistral(context)
            
        }
    for model in models:
        print(model)
        models[model].downloadModel()
        

if __name__ == '__main__':
    cli()