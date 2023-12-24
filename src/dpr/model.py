import torch
from torch import nn
from transformers import AutoModel

class Encoder(nn.Module):
    def __init__(self, 
                 model_checkpoint,
                 representation=0,
                 fixed=False):
        super(Encoder, self).__init__()

        self.encoder = AutoModel.from_pretrained(model_checkpoint)
        self.representation = representation
        self.fixed = fixed
        
    def get_representation(self, 
                          input_ids,
                          attention_mask,
                          token_type_ids=None):
        output = None
        if input_ids is not None:
            if self.fixed:
                with torch.no_grad():
                    outputs = self.encoder(input_ids,
                                           attention_mask, 
                                           token_type_ids)
            else:
                outputs = self.encoder(input_ids,
                                       attention_mask,
                                       token_type_ids)
            
            sequence_output = outputs['last_hidden_state']
            sequence_output = sequence_output.masked_fill(~attention_mask[..., None].bool(), 0.0)
        
            if self.representation > -2:
                output = sequence_output[:, self.representation, :]
            elif self.representation == -10:
                output = sequence_output.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            elif self.representation == -100:
                output = outputs[1]   
        return output
    
class BiEncoder(nn.Module):
    def __init__(self,
                 model_checkpoint,
                 q_encoder=None,
                 ctx_encoder=None,
                 representation=0,
                 q_fixed=False,
                 ctx_fixed=False):
        super(BiEncoder, self).__init__()
        if q_encoder == None:
            q_encoder = Encoder(model_checkpoint,
                                representation,
                                q_fixed)
        if ctx_encoder == None:
            ctx_encoder = Encoder(model_checkpoint,
                                  representation,
                                  ctx_fixed)
        self.q_encoder = q_encoder
        self.ctx_encoder = ctx_encoder

    def forward(self,
                q_ids,
                q_attn_mask,
                ctx_ids,
                ctx_attn_mask):
        q_out = self.q_encoder.get_representation(q_ids, q_attn_mask)
        ctx_out = self.ctx_encoder.get_representation(ctx_ids, ctx_attn_mask)

        return q_out, ctx_out

    def get_models(self):
        return self.q_encoder, self.ctx_encoder