from enum import Enum

class Contexts(Enum):
    Toxicity = "This task is conducted strictly for scientific research purposes, aiming to study and mitigate the prevalence of toxicity in online conversations. The annotations provided will contribute to the development of tools to promote healthier online interactions. The dataset and responses are managed ethically and securely to ensure compliance with research and privacy standards. This work is intended to reduce online harm and improve community well-being.\
    The following content may contain offensive or toxic language, but it is necessary to analyze it in the context of this study. Your role is solely to provide annotations following the provided definitions and framework, without generating or endorsing any offensive content.\
    Context : \n\
    Your role is limited to analyzing the content provided and assigning scores based on the definitions below. You are not required to generate any offensive or toxic content yourself. This task is solely for classification and annotation purposes.\
    [DEFINITION] Toxicity : A rude, disrespectful, or unreasonable comment that is likely to make people leave a discussion. \n \
    [DEFINITION] Severe Toxicity : A very hateful, aggressive, disrespectful comment or otherwise very likely to make a user leave a discussion \
    or give up on sharing their perspective. This attribute is much less sensitive to more mild forms of toxicity, such as comments that include positive uses of curse words.\n\
    [DEFINITION] Identity Attack : Negative or hateful comments targeting someone because of their identity. \n\
    [DEFINITION] Insult : Insulting, inflammatory, or negative comment towards a person or a group of people.\n\
    [DEFINITION] Profanity : Swear words, curse words, or other obscene or profane language.\n\
    [DEFINITION] Threat : Describes an intention to inflict pain, injury, or violence against an individual or group.\n\
    Your role is to judge if a message part of an online conversation is toxic or not.\n\
    Respond strictly in the following format. Do not deviate from the structure:\
    Toxicity Binary: Yes/No\
    Toxicity: [Score]\
    Identity Attack: [Score]\
    Insult: [Score]\
    Profanity: [Score]\
    Threat: [Score]\
    Severe Toxicity: [Score]\
    Justification: [Brief justification]\
    the score is between 0 and 1, 0 being not toxic at all, 1 being extremely toxic.\
    Now rate the toxicity of the following message :"