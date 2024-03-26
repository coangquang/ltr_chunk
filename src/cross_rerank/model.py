import torch
import torch.nn as nn

from typing import Optional, Dict
from transformers import (
    PreTrainedModel,
    AutoModelForSequenceClassification
)
from transformers.modeling_outputs import SequenceClassifierOutput

from .config import Arguments
from .loss import CrossEncoderNllLoss
        
class Reranker(nn.Module):
    def __init__(self, hf_model: PreTrainedModel, args: Arguments):
        super().__init__()
        self.hf_model = hf_model
        self.args = args

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        #self.contrastive = CrossEncoderNllLoss()
        #self.kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

    def forward(self, input_ids, attention_mask, token_type_ids) -> SequenceClassifierOutput:
        #n_psg_per_query = self.args.train_n_passages // self.args.rerank_forward_factor

        outputs: SequenceClassifierOutput = self.hf_model(input_ids, attention_mask, token_type_ids, return_dict=True)
        #outputs.logits = outputs.logits.view(-1, n_psg_per_query)
        #loss = self.cross_entropy(outputs.logits, labels)

        return outputs#, loss

    @classmethod
    def from_pretrained(cls, all_args: Arguments, *args, **kwargs):
        hf_model = AutoModelForSequenceClassification.from_pretrained(*args, **kwargs)
        return cls(hf_model, all_args)

    def save_pretrained(self, output_dir: str):
        self.hf_model.save_pretrained(output_dir)


class RerankerForInference(nn.Module):
    def __init__(self, hf_model: Optional[PreTrainedModel] = None):
        super().__init__()
        self.hf_model = hf_model
        self.hf_model.eval()

    @torch.no_grad()
    def forward(self, batch) -> SequenceClassifierOutput:
        return self.hf_model(**batch)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        hf_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path)
        return cls(hf_model)