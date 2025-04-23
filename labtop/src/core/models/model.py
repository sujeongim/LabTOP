import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config


class LabTOPModel(nn.Module):
    def __init__(self, config):
        super(LabTOPModel, self).__init__()
        self.hidden_size = config["model"]["hidden_size"]
        self.vocab_size = config["model"]["vocab_size"]
        self.num_layers = config["model"]["num_layers"]
        self.num_heads = config["model"]["num_heads"]
        self.dropout = config["model"]["dropout"]
        self.max_seq_len = config["model"]["max_seq_len"]
        self.activation_func = config["train"]["activation_func"]
        self.bos_token_id = config["train"]["bos_token_id"]
        self.eos_token_id = config["train"]["eos_token_id"]
        
        model_config = GPT2Config(
            activation_function=self.activation_func,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            n_embd=self.hidden_size,
            n_head=self.num_heads,
            n_layer=self.num_layers,
            n_positions=self.max_seq_len,
            embd_pdrop=self.dropout,
            resid_pdrop=self.dropout,
            vocab_size=self.vocab_size,
        )
        
        # Transformer-based encoder for task instructions
        self.model = GPT2LMHeadModel(model_config)
        
    def model_memory_usage(self):
        total_mem = 0
        total_n_params = 0
        for p in self.model.parameters():
            n_params = p.flatten().shape[0]
            if p.dtype == torch.float32:
                mem = 4 * n_params # bytes
            elif p.dtype == torch.float16:
                mem = 2 * n_params # bytes
            elif p.dtype == torch.uint8:
                mem = 1 * n_params # bytes
            else:
                raise ValueError()
            total_mem += mem
            total_n_params += n_params

        return total_mem / (1024**2)  # MiB
    
        
    def forward(self, input_ids, attention_mask=None):
        # input_ids: (batch_size, seq_len)
        # attention_mask: (batch_size, seq_len)
        outputs = self.model(
                input_ids = input_ids,
                attention_mask = attention_mask,
        )
        logits = outputs['logits']
        return logits

    def predict(self, input_ids, attention_mask=None):
        logits = self.forward(input_ids, attention_mask)
        return torch.argmax(logits, dim=-1)  # Predicted action sequence
    


