from enum import Enum

class Models(Enum):
    ChatGPT = 'ChatGPT'
    Perspective = 'Perspective'
    RoBERTa_ToxiGen = 'RoBERTa_ToxiGen'
    Hatebert_toxigen = 'Hatebert_toxigen'
    Llama = 'Llama'
    Claude = 'Claude'
    Gemini = 'Gemini'
    LlamaGuard = 'LlamaGuard'