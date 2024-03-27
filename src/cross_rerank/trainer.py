import os
import torch
from torch import nn
from torch.utils.checkpoint import get_device_states, set_device_states
#from typing import Optional, Union, Dict, Any
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.trainer import Trainer
from transformers.modeling_outputs import SequenceClassifierOutput
from collections.abc import Mapping
from .logger_config import logger
from .metrics import accuracy
from .utils import AverageMeter

def nested_detach(tensors):
    "Detach `tensors` (even if it's a nested list/tuple/dict of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_detach(t) for k, t in tensors.items()})
    return tensors.detach()

class RandContext:
    def __init__(self, *tensors):
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self):
        self._fork = torch.random.fork_rng(
            devices=self.fwd_gpu_devices,
            enabled=True
        )
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None
        
class RerankerTrainer(Trainer):
    def __init__(self, *pargs, **kwargs):
        super(RerankerTrainer, self).__init__(*pargs, **kwargs)

        self.acc_meter = AverageMeter('acc', round_digits=2)
        self.last_epoch = 0

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to {}".format(output_dir))

        self.model.save_pretrained(output_dir)

        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)
            
    def compute_loss(self, model, inputs, return_outputs=False):
        n_psg_per_query = self.args.train_n_passages // self.args.rerank_forward_factor
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        labels = inputs['labels']
        outputs = model(input_ids, attention_mask, token_type_ids)
        outputs.logits = outputs.logits.view(-1, n_psg_per_query)
        loss = self.model.cross_entropy(outputs.logits, labels)

        if self.model.training:
            step_acc = accuracy(output=outputs.logits.detach(), target=labels)[0]
            self.acc_meter.update(step_acc)
            if self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
                logger.info('step: {}, {}'.format(self.state.global_step, self.acc_meter))

            self._reset_meters_if_needed()

        return (loss, outputs) if return_outputs else loss
            
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss_train(model, inputs)

        return loss.detach() / self.args.gradient_accumulation_steps

    def compute_loss_train(self, model, inputs, return_outputs=False):
        #print(inputs)
        #print(inputs['input_ids'].size())
        n_psg_per_query = self.args.train_n_passages // self.args.rerank_forward_factor
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        labels = inputs['labels']
        
        all_reps, rnds = [], []
            
        id_chunks = input_ids.split(self.args.chunk_size)
        attn_mask_chunks = attention_mask.split(self.args.chunk_size)
            
        type_ids_chunks = token_type_ids.split(self.args.chunk_size)
            
        for id_chunk, attn_chunk, type_chunk in zip(id_chunks, attn_mask_chunks, type_ids_chunks):
            rnds.append(RandContext(id_chunk, attn_chunk, type_chunk))
            with torch.no_grad():
                chunk_reps = self.model(id_chunk, attn_chunk, type_chunk).logits
            all_reps.append(chunk_reps)
        all_reps = torch.cat(all_reps)
        all_reps = all_reps.view(-1, n_psg_per_query)
            
        all_reps = all_reps.float().detach().requires_grad_()
        loss = self.model.cross_entropy(all_reps, labels)
        
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss)
        #if self.args.gradient_accumulation_steps > 1:
        #    loss = loss / self.args.gradient_accumulation_steps
        #loss.backward()
        #temp = all_reps.view(-1,1)      
        grads = all_reps.grad.split(int(self.args.chunk_size/n_psg_per_query))
            
        for id_chunk, attn_chunk, type_chunk, grad, rnd in zip(id_chunks, attn_mask_chunks, type_ids_chunks, grads, rnds):
            #print(id_chunk.size())
            with rnd:
                chunk_reps = self.model(id_chunk, attn_chunk, type_chunk).logits
                #print(chunk_reps.size())
                #print(grad.size())
                surrogate = torch.dot(chunk_reps.flatten().float(), grad.flatten())
                
            self.accelerator.backward(surrogate)
                
        #outputs, loss = model(input_ids, attention_mask, token_type_ids, labels)

        if self.model.training:
            step_acc = accuracy(all_reps, target=labels)[0]
            #print(step_acc)
            self.acc_meter.update(step_acc)
            if self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
                logger.info('step: {}, {}'.format(self.state.global_step, self.acc_meter))

            self._reset_meters_if_needed()

        return (loss, all_reps) if return_outputs else loss
    
    '''def compute_loss_pred(self, model, inputs, return_outputs=False):
        #print(inputs)
        #print(inputs['input_ids'].size())
        n_psg_per_query = self.args.train_n_passages // self.args.rerank_forward_factor
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        labels = inputs['labels']
        
        all_reps, rnds = [], []
            
        id_chunks = input_ids.split(self.args.chunk_size)
        attn_mask_chunks = attention_mask.split(self.args.chunk_size)
            
        type_ids_chunks = token_type_ids.split(self.args.chunk_size)
            
        for id_chunk, attn_chunk, type_chunk in zip(id_chunks, attn_mask_chunks, type_ids_chunks):
            rnds.append(RandContext(id_chunk, attn_chunk, type_chunk))
            with torch.no_grad():
                chunk_reps = self.model(id_chunk, attn_chunk, type_chunk).logits
            all_reps.append(chunk_reps)
        all_reps = torch.cat(all_reps)
        all_reps = all_reps.view(-1, n_psg_per_query)
        loss = self.model.cross_entropy(all_reps, labels)

        if self.model.training:
            step_acc = accuracy(all_reps, target=labels)[0]
            #print(step_acc)
            self.acc_meter.update(step_acc)
            if self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
                logger.info('step: {}, {}'.format(self.state.global_step, self.acc_meter))

            self._reset_meters_if_needed()

        return (loss, all_reps) if return_outputs else loss
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()

                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)'''

    def _reset_meters_if_needed(self):
        if int(self.state.epoch) != self.last_epoch:
            self.last_epoch = int(self.state.epoch)
            self.acc_meter.reset()