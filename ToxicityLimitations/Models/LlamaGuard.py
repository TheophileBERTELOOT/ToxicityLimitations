from transformers import AutoTokenizer, AutoModelForCausalLM
import torch 

class LlamaGuard:
    def __init__(self) -> None:
        model_id = "meta-llama/Llama-Guard-3-8B"
        self.device = "cuda"
        dtype = torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=self.device)

    
    def moderate(self,chat):
        input_ids = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to(self.device)
        output = self.model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
        prompt_len = input_ids.shape[-1]
        return self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
    
    def getToxicityScore(self,message):
        result = self.moderate(self.context+'\n'+message)
        return result
    
    def standardizeOutput(self):
        pass




