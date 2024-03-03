import logging
import os

import torch
from torch import nn
from transformers import BertForMaskedLM, AutoModelForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput

from arguments import ModelArguments
from enhancedDecoder import BertLayerForDecoder, DebertaV2LayerForDecoder

logger = logging.getLogger(__name__)


class RetroMAEForPretraining(nn.Module):
    def __init__(
            self,
            bert,
            model_args: ModelArguments,
    ):
        super(RetroMAEForPretraining, self).__init__()
        self.lm = bert
        if hasattr(self.lm, 'deberta'):
            self.decoder_embeddings = self.lm.deberta.embeddings
            self.c_head = DebertaV2LayerForDecoder(bert.config)
            self.model_type = 'deberta-v2'
        elif hasattr(self.lm, 'bert'):
            self.decoder_embeddings = self.lm.bert.embeddings
            self.c_head = BertLayerForDecoder(bert.config)
            self.model_type = 'bert'
        elif hasattr(self.lm, 'roberta'):
            self.decoder_embeddings = self.lm.roberta.embeddings
            self.c_head = BertLayerForDecoder(bert.config)
            self.model_type = 'roberta'
        else:
            self.decoder_embeddings = self.lm.bert.embeddings
            self.c_head = BertLayerForDecoder(bert.config)
            self.model_type = 'non'

        
        self.c_head.apply(self.lm._init_weights)

        self.cross_entropy = nn.CrossEntropyLoss()

        self.model_args = model_args

    def gradient_checkpointing_enable(self, **kwargs):
        self.lm.gradient_checkpointing_enable(**kwargs)

    def forward(self,
                encoder_input_ids, encoder_attention_mask, encoder_labels,
                decoder_input_ids, decoder_attention_mask, decoder_labels):

        lm_out: MaskedLMOutput = self.lm(
            encoder_input_ids, encoder_attention_mask,
            labels=encoder_labels,
            output_hidden_states=True,
            return_dict=True
        )
        if self.model_args.representation == 'cls':
            cls_hiddens = lm_out.hidden_states[-1][:, :1]  # B 1 D
        elif self.model_args.representation == 'mean':
            print(encoder_attention_mask.size())
            sequence_output = lm_out.hidden_states[-1]
            print(sequence_output.size())
            s = torch.sum(sequence_output * encoder_attention_mask.unsqueeze(-1).float(), dim=1)
            print(s.size())
            d = encoder_attention_mask.sum(axis=1, keepdim=True).float()
            print(d.size())
            cls_hiddens = s / d
            print(sequence_output.size())
            cls_hiddens = torch.nn.functional.normalize(cls_hiddens, dim=-1)
            print(sequence_output.size())

        decoder_embedding_output = self.decoder_embeddings(input_ids=decoder_input_ids)
        hiddens = torch.cat([cls_hiddens, decoder_embedding_output[:, 1:]], dim=1)
        if self.model_type == 'bert':
            decoder_position_ids = self.lm.bert.embeddings.position_ids[:, :decoder_input_ids.size(1)]
            #print(decoder_position_ids.size())
            decoder_position_embeddings = self.lm.bert.embeddings.position_embeddings(decoder_position_ids)  # B L D
            #print(decoder_position_embeddings.size())
            query = decoder_position_embeddings + cls_hiddens
            #print(query.size())
        elif self.model_type == 'roberta':
            decoder_position_ids = self.lm.roberta.embeddings.position_ids[:, :decoder_input_ids.size(1)]
            #print(decoder_position_ids.size())
            decoder_position_embeddings = self.lm.roberta.embeddings.position_embeddings(decoder_position_ids)  # B L D
            #print(decoder_position_embeddings.size())
            query = decoder_position_embeddings + cls_hiddens
            #print(query.size())
        elif self.model_type == 'deberta-v2':
            query = torch.cat([cls_hiddens for u in range(hiddens.size(1))], dim=1)
            #print(query.size())

        # cls_hiddens = cls_hiddens.expand(hiddens.size(0), hiddens.size(1), hiddens.size(2))
        # query = self.decoder_embeddings(inputs_embeds=cls_hiddens)

        matrix_attention_mask = self.lm.get_extended_attention_mask(
            decoder_attention_mask,
            decoder_attention_mask.shape,
            decoder_attention_mask.device
        )
        
        #print(matrix_attention_mask.size())
        #print(decoder_labels.size())
        if self.model_type != 'deberta-v2':
            hiddens = self.c_head(query=query,
                                key=hiddens,
                                value=hiddens,
                                attention_mask=matrix_attention_mask)[0]
        else:
            hiddens = self.c_head(query=query,
                                key=hiddens,
                                value=hiddens,
                                attention_mask=matrix_attention_mask)
        
        #print(hiddens.size())
        
        pred_scores, loss = self.mlm_loss(hiddens, decoder_labels)

        return (loss + lm_out.loss,)

    def mlm_loss(self, hiddens, labels):
        if hasattr(self.lm, 'cls'):
            pred_scores = self.lm.cls(hiddens)
        elif hasattr(self.lm, 'lm_head'):
            pred_scores = self.lm.lm_head(hiddens)
        else:
            raise NotImplementedError

        masked_lm_loss = self.cross_entropy(
            pred_scores.view(-1, self.lm.config.vocab_size),
            labels.view(-1)
        )
        return pred_scores, masked_lm_loss

    def save_pretrained(self, output_dir: str):
        self.lm.save_pretrained(os.path.join(output_dir, "encoder_model"))
        torch.save(self.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments,
            *args, **kwargs
    ):
        hf_model = AutoModelForMaskedLM.from_pretrained(*args, **kwargs)
        model = cls(hf_model, model_args)
        return model