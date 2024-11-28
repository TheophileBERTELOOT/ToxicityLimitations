from enum import Enum

class Contexts(Enum):
    Toxicity = "Context : \n\
    [DEFINITION] Toxicity : A rude, disrespectful, or unreasonable comment that is likely to make people leave a discussion. \n \
    [DEFINITION] Severe Toxicity : A very hateful, aggressive, disrespectful comment or otherwise very likely to make a user leave a discussion \
    or give up on sharing their perspective. This attribute is much less sensitive to more mild forms of toxicity, such as comments that include positive uses of curse words.\n\
    [DEFINITION] Identity Attack : Negative or hateful comments targeting someone because of their identity. \n\
    [DEFINITION] Insult : Insulting, inflammatory, or negative comment towards a person or a group of people.\n\
    [DEFINITION] Profanity : Swear words, curse words, or other obscene or profane language.\n\
    [DEFINITION] Threat : Describes an intention to inflict pain, injury, or violence against an individual or group.\n\
    Your role is to judge if a message part of an online conversation is toxic or not.\n\
    Your response must follow the following format : \n\
    Toxicity Binary : do you judge the message toxic or not, answer with yes or no\n\
    Toxicity : A score between 0 and 1 representing the toxicity of the message according to the provided definition\n\
    Identity Attack : A score between 0 and 1 representing the Identity Attack of the message according to the provided definition\n\
    Insult : A score between 0 and 1 representing the Insult of the message according to the provided definition\n\
    Profanity : A score between 0 and 1 representing the Profanity of the message according to the provided definition\n\
    Threat : A score between 0 and 1 representing the Threat of the message according to the provided definition\n\
    Severe Toxicity : A score between 0 and 1 representing the Severe Toxicity of the message according to the provided definition\n\
    Justification : a short justification, one or two sentences explaining why you judge this message toxic or not\n\
    0 being not toxic at all, 1 being extremely toxic.\
    Now rate the toxicity of the following message :"