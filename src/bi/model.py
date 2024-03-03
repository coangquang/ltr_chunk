import torch
from torch import nn
from transformers import AutoModel

class Encoder(nn.Module):
    def __init__(self, 
                 model_checkpoint,
                 representation='cls',
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
            #sequence_output = sequence_output.masked_fill(~attention_mask[..., None].bool(), 0.0)
        
            if self.representation == 'cls':
                output = sequence_output[:, 0, :]
            elif self.representation == 'mean':
                s = torch.sum(sequence_output * attention_mask.unsqueeze(-1).float(), dim=1)
                d = attention_mask.sum(axis=1, keepdim=True).float()
                output = s / d
                output = torch.nn.functional.normalize(output, dim=-1)
                #output = sequence_output.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            #elif self.representation == -100:
            #    output = outputs[1]   
        return output
    
    def save(self, output_dir: str):
        state_dict = self.encoder.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
                 v in state_dict.items()})
        self.encoder.save_pretrained(output_dir, state_dict=state_dict)
    
class SharedBiEncoder(nn.Module):
    def __init__(self,
                 model_checkpoint,
                 encoder=None,
                 representation='cls',
                 fixed=False):
        super(SharedBiEncoder, self).__init__()
        if encoder == None:
            encoder = Encoder(model_checkpoint,
                              representation,
                              fixed)

        self.encoder = encoder

    def forward(self,
                q_ids,
                q_attn_mask,
                ctx_ids,
                ctx_attn_mask):
        q_out = self.encoder.get_representation(q_ids, q_attn_mask)
        ctx_out = self.encoder.get_representation(ctx_ids, ctx_attn_mask)

        return q_out, ctx_out

    def get_model(self):
        return self.encoder