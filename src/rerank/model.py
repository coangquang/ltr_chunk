import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

class CrossEncoder(nn.Module):
    def __init__(self, 
                 encoder= None,
                 model_checkpoint="vinai/phobert-base-v2",
                 representation=0,
                 dropout=0.1,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(CrossEncoder, self).__init__()
        if encoder:
            self.encoder=encoder
        else:
            self.encoder = AutoModel.from_pretrained(model_checkpoint)
        self.representation = representation
        self.classifier = nn.Sequential(nn.Dropout(p=dropout),
                                        nn.Linear(self.encoder.config.hidden_size,2))
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.device = device
        self.sep_token = " . " + self.tokenizer.sep_token + " "

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None):
        
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
            
        logits = self.classifier(output)
            
        return logits
    
    def predict(self, cross_samples, max_len):
        cross_texts = [q + self.sep_token + p for (q,p) in cross_samples]
        C = self.tokenizer.batch_encode_plus(cross_texts,
                                             padding='max_length',
                                             truncation=True,
                                             max_length=max_len,
                                             return_tensors='pt')
        with torch.no_grad():
            logits = self.forward(C['input_ids'].to(self.device),
                            C['attention_mask'].to(self.device))  
            probs = nn.functional.softmax(logits, dim=-1)
        return probs