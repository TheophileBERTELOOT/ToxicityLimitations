from enum import Enum

class Models(Enum):
    ChatGPT = 'ChatGPT'
    Perspective = 'Perspective'
    RoBERTa_ToxiGen = 'RoBERTa_ToxiGen'
    Hatebert_toxigen = 'Hatebert_toxigen'
    Llama = 'Llama'#ollama
    Claude = 'Claude'
    Gemini = 'Gemini'
    LlamaGuard = 'LlamaGuard'#ollama
    Falcon = 'Falcon'#ollama
    Mistral = 'Mistral' #ollama
    Gemma2 = 'gemma2' #ollama
    Qwen25 = 'qwen'#ollama
    phi35 = 'phi35'#ollama
    dolphin_llama3 = 'dolphin_llama3'#ollama
    mistral_openorca = 'mistral-openorca'#ollama
    aya = 'aya'#ollama
    openhermes = 'openhermes'#ollama
    neural_chat = 'neural-chat'#ollama
    llama3_chatqa = 'llama3-chatqa'#ollama
    hermes3 = 'hermes3'#ollama
    granite3_guardian = 'granite3-guardian'#ollama