import click
import configparser
from ToxicityLimitations.Models.Models import Models
from ToxicityLimitations.Datasets.Datasets import Datasets
from ToxicityLimitations.Experiments.Experiments import Experiments
from ToxicityLimitations.Models.Contexts import Contexts

# from ToxicityLimitations.Models.chatGPT import chatGPTVanilla
from ToxicityLimitations.Models.RobertaToxicGen import RoBERTa_ToxiGen
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

def cli(model,message,config_path,datasets,experiments,output_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    openai_api_key = config['openAI']['api_key']
    org_key = config['openAI']['org']
    project_key = config['openAI']['project']
    perspective_api_key = config['PerspectiveAPI']['api_key']
    gemini_api_key = config['Gemini']['api_key']
    claude_api_key = config['Claude']['api_key']
    mistral_api_key = config['Mistral']['api_key']

    context = Contexts.Toxicity.value
    models = {
            # Models.aya.value : Aya(context),
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
            Models.RoBERTa_ToxiGen.value : RoBERTa_ToxiGen(),
            Models.Hatebert_toxigen.value : Hatebert_toxigen(),
            # Models.Perspective.value : Perspective(perspective_api_key),
            # Models.Gemini.value : Gemini(gemini_api_key,context),
            # Models.Claude.value : Claude(claude_api_key,context),
            # Models.Llama.value : Llama(context),
            # Models.LlamaGuard.value: LlamaGuard(context)
            # Models.Falcon.value : Falcon(context),
            # Models.Mistral.value : Mistral(context)
            
        }
    
    if experiments == Experiments.Subtle.value:
        xp = SubtleExperiments(models,output_path)
        xp.run()
    if experiments == Experiments.Toxigen.value:
        xp = ToxigenExperiments(models,output_path)
        xp.run()
        
        
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
        m = Gemini(gemini_api_key)
        m.getToxicityScore(message)
    elif model == Models.Llama.value:
        m = Llama()
        m.getToxicityScore(message)
    elif model == Models.LlamaGuard.value:
        m = LlamaGuard()
        m.getToxicityScore(message)
        

if __name__ == '__main__':
    cli()