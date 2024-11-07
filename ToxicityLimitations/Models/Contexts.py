from enum import Enum

class Contexts(Enum):
    Toxicity = "[DEFINITION] Toxicity : A rude, disrespectful, or unreasonable comment that is likely to make people leave a discussion. \n \
    [DEFINITION] Severe Toxicity : A very hateful, aggressive, disrespectful comment or otherwise very likely to make a user leave a discussion \
    or give up on sharing their perspective. This attribute is much less sensitive to more mild forms of toxicity, such as comments that include positive uses of curse words.\n\
    [DEFINITION] Identity attack : Negative or hateful comments targeting someone because of their identity. \n\
    [DEFINITION] Insult : Insulting, inflammatory, or negative comment towards a person or a group of people.\n\
    [DEFINITION] Profanity : Swear words, curse words, or other obscene or profane language.\n\
    [DEFINITION] Threat : Describes an intention to inflict pain, injury, or violence against an individual or group.\n\
    Your role is to judge if a message part of an online conversation is toxic or not.\n\
    Your response must follow the following format : \n\
    Toxic Binary : do you judge the message toxic or not, answer with yes or no\n\
    Toxicity : A score between 0 and 1 representing the toxicity of the message according to the provided definition\n\
    Identity attack : A score between 0 and 1 representing the toxicity of the message according to the provided definition\n\
    Insult : A score between 0 and 1 representing the toxicity of the message according to the provided definition\n\
    Profanity : A score between 0 and 1 representing the toxicity of the message according to the provided definition\n\
    Threat : A score between 0 and 1 representing the toxicity of the message according to the provided definition\n\
    Severe Toxicity : A score between 0 and 1 representing the toxicity of the message according to the provided definition\n\
    Justification : a short justification, one or two sentences explaining why you judge this message toxic or not"