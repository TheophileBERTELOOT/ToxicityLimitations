import click
import configparser
from ToxicityLimitations.Models.Models import Models
from ToxicityLimitations.Datasets.Datasets import Datasets

from ToxicityLimitations.Models.chatGPT import chatGPTVanilla
from ToxicityLimitations.Models.RobertaToxicGen import RoBERTa_ToxiGen
from ToxicityLimitations.Models.HateBertToxicGen import Hatebert_toxigen
from ToxicityLimitations.Models.Perspective import Perspective

from ToxicityLimitations.Datasets.Subtle import SubtleDataset

@click.command(help="Code for the article 'Limitations of modern toxicity detection models' ")
@click.option(
    '--model', '-m', 
    type=click.Choice([m.value for m in Models]+['None'], case_sensitive=False),
    default='None', 
    help='The model used to check the toxicity of a message',
    prompt=True
)

@click.option(
    '--message', '-msg', 
    type=str,
    default='Bonjour !!!!!!!', 
    help='the message to check the toxicity off',
)

@click.option(
    '--config-path', '-cfg', 
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default='../config.ini', 
    help='the path of the config file with the api keys',
)

@click.option(
    '--datasets', '-d', 
    type=click.Choice([m.value for m in Datasets]+['None'], case_sensitive=False),
    default='None', 
    help='The dataset used to compare the models',
    prompt=True
)

def cli(model,message,config_path,datasets):
    config = configparser.ConfigParser()
    config.read(config_path)
    
    if datasets == Datasets.Subtle.value:
        dataset = SubtleDataset()
    
    if model == Models.ChatGPT.value:
        openai_api_key = config['openAI']['api_key']
        org_key = config['openAI']['org']
        project_key = config['openAI']['project']
        m = chatGPTVanilla(openai_api_key,org_key,project_key)
        m.getToxicityScore(message)
    elif model == Models.RoBERTa_ToxiGen.value:
        m = RoBERTa_ToxiGen()
        m.getToxicityScore(message)
    elif model == Models.Hatebert_toxigen.value:
        m = Hatebert_toxigen()
        m.getToxicityScore(message)
    elif model == Models.Perspective.value:
        perspective_api_key = config['PerspectiveAPI']['api_key']
        m = Perspective(perspective_api_key)
        m.getToxicityScore()
        
        

if __name__ == '__main__':
    cli()