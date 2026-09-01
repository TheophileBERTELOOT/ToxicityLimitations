import sys
sys.path.append(".")
sys.path.append("../")

import click
import configparser
from huggingface_hub import login
from Models.Models import Models
from Datasets.Datasets import Datasets
from Experiments.Experiments import Experiments
from Models.Contexts import Contexts


# from Models.chatGPT import chatGPTVanilla
# from Models.RobertaToxicGen import RoBERTa_ToxiGen
# from Models.HateBertToxicGen import Hatebert_toxigen
# # from Models.Perspective import Perspective
# from Models.Llama import Llama
# from Models.LlamaGuard import LlamaGuard
# # from Models.Gemini import Gemini
# # from Models.claude import Claude
# from Models.Falcon import Falcon
# from Models.Mistral import Mistral
# from Models.Aya import Aya
# from Models.DolphinLlama import DolphinLlama
# from Models.Gemma2 import Gemma2
# from Models.Granite3Guardian import Granite3Guardian
# from Models.Hermes3 import Hermes3
# from Models.Llama3Chat import Llama3ChatQA
# from Models.MistralOrca import MistralOpenOrca
# from Models.NeuralChat import NeuralChat
# from Models.OpenHermes import Openhermes
# # from Models.Phi35 import Phi35
# from Models.Qwen25 import Qwen25


# from Datasets.Subtle import SubtleDataset

# from Experiments.Subtle import SubtleExperiments
from Experiments.Toxigen import ToxigenExperiments
from Experiments.Attack import AttackExperiments
# from Experiments.Conversation import ConversationExperiments
from Experiments.Wikipedia import WikipediaExperiments
from Experiments.Biais import BiaisExperiments

@click.command(help="Code for the article 'Limitations of modern toxicity detection models' ")
@click.option(
    '--model', '-m', 
    type=click.Choice([m.value for m in Models]+['None'], case_sensitive=False),
    default='None', 
    help='The model used to check the toxicity of a message',
    # prompt=True,
    required = False
)

@click.option(
    '--message', '-msg', 
    type=str,
    default='Bonjour !!!!!!!', 
    help='the message the models have to check',
    required = False
)

@click.option(
    '--config-path', '-cfg', 
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default='../config.ini', 
    help='the path of the config file with the api keys',
    required = False
)

@click.option(
    '--datasets', '-d', 
    type=click.Choice([m.value for m in Datasets]+['None'], case_sensitive=False),
    default='None', 
    help='The dataset used to compare the models',
    # prompt=True,
    required = False
)

@click.option(
    '--experiments', '-xp', 
    type=click.Choice([m.value for m in Experiments]+['None'], case_sensitive=False),
    default='None', 
    help='The experiment to run',
    # prompt=True,
    required = False
)

@click.option(
    '--output-path', '-out', 
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default='../results/', 
    help='the path where the experiments results should be store',
    required = False
)

@click.option(
    '--compute-results', '-cr', 
    type=bool,
    default=False, 
    help='Do you want to compute the results of the experience',
    required = False
)

def cli(model,message,config_path,datasets,experiments,output_path,compute_results):
    config = configparser.ConfigParser()
    config.read(config_path)
#    login(config['HuggingFace']['token'])
    openai_api_key = config['openAI']['api_key']
    org_key = config['openAI']['org']
    project_key = config['openAI']['project']
    perspective_api_key = config['PerspectiveAPI']['api_key']
    gemini_api_key = config['Gemini']['api_key']
    claude_api_key = config['Claude']['api_key']
    mistral_api_key = config['Mistral']['api_key']
    models_path = "/lustre06/project/6010878/thber64/ToxicityLimitationsProject/ToxicityLimitations/media/theophileberteloot/ad5f5ae6-6fd8-46ef-8659-a07d195c6d70/PhD/ToxicityLimitations/Models/"

    context = Contexts.Toxicity.value
    models = {
            # Models.aya.value : Aya(context,isOffline=True,model_path=models_path),
            # Models.dolphin_llama3.value : DolphinLlama(context,isOffline=True,model_path=models_path),
            # Models.Gemma2.value  : Gemma2(context,isOffline=True,model_path=models_path),
            # Models.granite3_guardian.value : Granite3Guardian(context,isOffline=True,model_path=models_path),
            # Models.Qwen25.value : Qwen25(context,isOffline=True,model_path=models_path),
            # Models.hermes3.value : Hermes3(context,isOffline=True,model_path=models_path),
            # Models.llama3_chatqa.value : Llama3ChatQA(context,isOffline=True,model_path=models_path),
            # Models.mistral_openorca.value: MistralOpenOrca(context,isOffline=True,model_path=models_path),
            # Models.neural_chat.value : NeuralChat(context,isOffline=True,model_path=models_path),
            # Models.openhermes.value : Openhermes(context,isOffline=True,model_path=models_path),
            # Models.phi35.value : Phi35(context,isOffline=True,model_path=models_path),
            Models.ChatGPT.value : 'ChatGPT',#chatGPTVanilla(openai_api_key,org_key,project_key,context),
            # Models.RoBERTa_ToxiGen.value : RoBERTa_ToxiGen(isOffline=True,model_path=models_path),
            # Models.Hatebert_toxigen.value : Hatebert_toxigen(isOffline=True,model_path=models_path),
            #Models.Perspective.value : Perspective(perspective_api_key),
            # Models.Gemini.value : Gemini(gemini_api_key,context),
            # Models.Claude.value : Claude(claude_api_key,context),
            # Models.Llama.value : Llama(context,isOffline=True,model_path=models_path),
            # Models.LlamaGuard.value: LlamaGuard(context,isOffline=True,model_path=models_path),
            # Models.Falcon.value : Falcon(context,isOffline=True,model_path=models_path),
            # Models.Mistral.value : Mistral(context,isOffline=True,model_path=models_path)
            
        }
    


    if not compute_results:
        if experiments == Experiments.Subtle.value:
            xp = SubtleExperiments(models,output_path)
            xp.run()
        if experiments == Experiments.Toxigen.value:
            xp = ToxigenExperiments(models,output_path)
            xp.run(open_api_key=openai_api_key,org=org_key,project=project_key)
        if experiments == Experiments.Wikipedia.value:
            xp = WikipediaExperiments(models,output_path)
            xp.run(open_api_key=openai_api_key,org=org_key,project=project_key)
        if experiments == Experiments.Conversation.value:
            xp = ConversationExperiments(models,output_path)
            xp.run()
        if experiments == Experiments.Attack.value:
            xp = AttackExperiments(models,output_path)
            xp.run()
        if experiments == Experiments.Biais.value:
            xp = BiaisExperiments(models,output_path)
            xp.run()
    else:
        if experiments == Experiments.Subtle.value:
            xp = SubtleExperiments(models,output_path)
            xp.compute_results()
        if experiments == Experiments.Toxigen.value:
            xp = ToxigenExperiments(models,output_path)
            xp.compute_results() 
        if experiments == Experiments.Wikipedia.value:
            xp = WikipediaExperiments(models,output_path)
            xp.compute_results() 
        if experiments == Experiments.Biais.value:
            xp = BiaisExperiments(models,output_path)
            xp.compute_results() 
        if experiments == Experiments.Attack.value:
            xp = AttackExperiments(models,output_path)
            xp.compute_results() 
        
        
    if datasets == Datasets.Subtle.value:
        dataset = SubtleDataset()
    
    if model == Models.ChatGPT.value:
        m = chatGPTVanilla(openai_api_key,org_key,project_key,context)
        m.getToxicityScore(message)
    elif model == Models.RoBERTa_ToxiGen.value:
        m = RoBERTa_ToxiGen()
        m.getToxicityScore(message)
    elif model == Models.Hatebert_toxigen.value:
        m = Hatebert_toxigen()
        m.getToxicityScore(message)
    elif model == Models.Perspective.value:
        m = Perspective(perspective_api_key)
        m.getToxicityScore(message)
    elif model == Models.Gemini.value:
        m = Gemini(gemini_api_key,context)
        m.getToxicityScore(message)
    elif model == Models.Llama.value:
        m = Llama(context)
        m.getToxicityScore(message)
    elif model == Models.LlamaGuard.value:
        m = LlamaGuard(context)
        m.getToxicityScore(message)
    elif model == Models.aya.value:
        m = Aya(context,isOffline=True)
        m.getToxicityScore(message)
        

if __name__ == '__main__':
    cli()
