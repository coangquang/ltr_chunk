import time
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.checkpoint import get_device_states, set_device_states
from torch.nn import DataParallel
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import LambdaLR

from .model import BiEncoder
from .loss import BiEncoderNllLoss, BiEncoderDoubleNllLoss


class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))

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
    
class DPRTrainer():
    """
    Trainer for biencoder
    """
    def __init__(self,
                 args,
                 train_loader, 
                 val_loader):
        self.parallel = True if torch.cuda.device_count() > 1 else False
        print("No of GPU(s):",torch.cuda.device_count())
        
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BiEncoder(model_checkpoint=self.args.BE_checkpoint,
                               representation=self.args.BE_representation,
                               q_fixed=self.args.q_fixed,
                               ctx_fixed=self.args.ctx_fixed)
        if self.args.load_path is not None:
            self.model.load_state_dict(torch.load(self.args.load_path))
        if self.parallel:
            print("Parallel Training")
            self.model = DataParallel(self.model)
        self.model.to(self.device)
        if self.args.BE_loss == 0.0 or self.args.BE_loss == 1.0:
            self.criterion = BiEncoderNllLoss(score_type=self.args.BE_score)
        else:
            self.criterion = BiEncoderDoubleNllLoss(score_type=self.args.BE_score,
                                                    alpha=self.args.BE_loss)
            
        self.val_criterion = BiEncoderNllLoss(score_type=self.args.BE_score)
        self.optimizer = AdamW(self.model.parameters(), lr=args.BE_lr) 
        self.scheduler = WarmupLinearSchedule(self.optimizer, 0.1 * len(self.train_loader) * self.args.BE_num_epochs, len(self.train_loader) * self.args.BE_num_epochs)
        self.epoch = 0
        self.patience_counter = 0
        self.best_val_acc = 0.0
        self.epochs_count = []
        self.train_losses = []
        self.valid_losses = []
        self.train_acc = []
        self.valid_acc = []
        
    def train_biencoder(self):
        # Compute loss and accuracy before starting (or resuming) training.
        print("\n",
              20 * "=",
              "Validation before training",
              20 * "=")
        val_time, val_loss, val_acc = self.validate()
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n"
              .format(val_time, val_loss, (val_acc*100)))
        print("\n",
              20 * "=",
              "Training biencoder model on device: {}".format(self.device),
              20 * "=")
        while self.epoch < self.args.BE_num_epochs:
            self.epoch +=1
            self.epochs_count.append(self.epoch)
            print("* Training epoch {}:".format(self.epoch))
            epoch_avg_loss, epoch_accuracy, epoch_time = self.train()
            self.train_losses.append(epoch_avg_loss)
            self.train_acc.append(epoch_accuracy.to('cpu')*100)
            print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%"
                  .format(epoch_time, epoch_avg_loss, (epoch_accuracy*100)))
        
            print("* Validation for epoch {}:".format(self.epoch))
            epoch_time, epoch_loss, epoch_accuracy = self.validate()
            self.valid_losses.append(epoch_loss)
            self.valid_acc.append(epoch_accuracy.to('cpu')*100)
            print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n"
                  .format(epoch_time, epoch_loss, (epoch_accuracy*100)))

            if epoch_accuracy <= self.best_val_acc:
                self.patience_counter += 1

            else:
                self.best_val_acc = epoch_accuracy
                self.patience_counter = 0
                if self.parallel:
                    torch.save(self.model.module.state_dict(), self.args.biencoder_path)
                else:
                    torch.save(self.model.state_dict(), self.args.biencoder_path)
        
            #if self.args.BE_num_epochs >= 5 and self.epoch % int(self.args.BE_num_epochs*0.2) == 0:
            if self.epoch == self.args.BE_num_epochs:
                if self.parallel:
                    #torch.save(self.model.module.state_dict(),
                    #           "hard{}_epoch{}_batch{}_ratio{}.pth.tar".format(self.args.no_hard,
                    #                                                           self.args.BE_num_epochs,
                    #                                                           self.args.BE_train_batch_size,
                    #                                                           self.args.BE_loss))
                    torch.save(self.model.module.state_dict(), self.args.final_path)
                    
                else:
                    #torch.save(self.model.state_dict(),
                    #           "hard{}_epoch{}_batch{}_ratio{}.pth.tar".format(self.args.no_hard,
                    #                                                           self.args.BE_num_epochs,
                    #                                                           self.args.BE_train_batch_size,
                    #                                                           self.args.BE_loss))
                    torch.save(self.model.state_dict(), self.args.final_path)
                    
                    
                #else:
                 #   torch.save(self.model.state_dict(), "epoch{}.pth.tar".format(self.epoch))
            
        # Plotting of the loss curves for the train and validation sets.
        plt.figure()
        plt.plot(self.epochs_count, self.train_losses, "-r")
        plt.plot(self.epochs_count, self.valid_losses, "-b")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend(["Training loss", "Validation loss"])
        plt.title("Cross entropy loss")
        plt.show()
    
        plt.figure()
        plt.plot(self.epochs_count, self.train_acc, '-r')
        plt.plot(self.epochs_count, self.valid_acc, "-b")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend(["Training accuracy", "Validation accuracy"])
        plt.title("Accuracy")
        plt.show()
    
        #return the final q_model, ctx_model
        if self.parallel:
            return self.model.module.get_models()
        else:
            return self.model.get_models()
        
    def train(self):
        self.model.train()
        epoch_start = time.time()
        batch_time_avg = 0.0
        epoch_loss = 0.0
        epoch_correct = 0
        tqdm_batch_iterator = tqdm(self.train_loader)
        for i, batch in enumerate(tqdm_batch_iterator):
            batch_start = time.time()
            if self.args.grad_cache:
                loss, num_correct = self.step_cache(batch)
            else:
                loss, num_correct = self.step(batch)
            batch_time_avg += time.time() - batch_start
            epoch_loss += loss
            epoch_correct += num_correct

            description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}"\
                        .format(batch_time_avg/(i+1),
                        epoch_loss/(i+1))
            tqdm_batch_iterator.set_description(description)

        epoch_time = time.time() - epoch_start
        epoch_avg_loss = epoch_loss / len(self.train_loader)
        epoch_accuracy = epoch_correct / len(self.train_loader.dataset)

        return epoch_avg_loss, epoch_accuracy, epoch_time
    
    def step(self, batch):
        self.model.train()
        if self.args.no_hard != 0:
            q_input_ids, q_attn_mask, p_input_ids, p_attn_mask, n_input_ids, n_attn_mask = tuple(t.to(self.device) for t in batch)
            ctx_len = n_input_ids.size()[-1]
            n_input_ids = n_input_ids.view(-1,ctx_len)
            n_attn_mask = n_attn_mask.view(-1,ctx_len)
            ctx_input_ids = torch.cat((p_input_ids, n_input_ids), 0)
            ctx_attn_mask = torch.cat((p_attn_mask, n_attn_mask), 0)
        else:
            q_input_ids, q_attn_mask, ctx_input_ids, ctx_attn_mask = tuple(t.to(self.device) for t in batch)
        
        self.optimizer.zero_grad()

        q_vectors, ctx_vectors = self.model(q_input_ids, q_attn_mask, ctx_input_ids, ctx_attn_mask)
        loss, num_correct = self.criterion.calc(q_vectors, ctx_vectors)
        loss.backward()

        self.optimizer.step()
        self.scheduler.step()

        return loss.item(), num_correct
    
    def step_cache(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        if self.args.no_hard != 0:
            q_input_ids, q_attn_mask, p_input_ids, p_attn_mask, n_input_ids, n_attn_mask = tuple(t.to(self.device) for t in batch)
            
            
            ctx_len = n_input_ids.size()[-1]
            n_input_ids = n_input_ids.view(-1,ctx_len)
            n_attn_mask = n_attn_mask.view(-1,ctx_len)
            ctx_input_ids = torch.cat((p_input_ids, n_input_ids), 0)
            ctx_attn_mask = torch.cat((p_attn_mask, n_attn_mask), 0)
        
        else:
            q_input_ids, q_attn_mask, ctx_input_ids, ctx_attn_mask = tuple(t.to(self.device) for t in batch)
            
        all_q_reps, all_ctx_reps = [], []
        q_rnds, ctx_rnds = [], []
            
        q_id_chunks = q_input_ids.split(self.args.q_chunk_size)
        q_attn_mask_chunks = q_attn_mask.split(self.args.q_chunk_size)
            
        ctx_id_chunks = ctx_input_ids.split(self.args.ctx_chunk_size)
        ctx_attn_mask_chunks = ctx_attn_mask.split(self.args.ctx_chunk_size)
            
        for id_chunk, attn_chunk in zip(q_id_chunks, q_attn_mask_chunks):
            q_rnds.append(RandContext(id_chunk, attn_chunk))
            with torch.no_grad():
                q_chunk_reps = self.model(id_chunk, attn_chunk, None, None)[0]
            all_q_reps.append(q_chunk_reps)
        all_q_reps = torch.cat(all_q_reps)
            
        for id_chunk, attn_chunk in zip(ctx_id_chunks, ctx_attn_mask_chunks):
            ctx_rnds.append(RandContext(id_chunk, attn_chunk))
            with torch.no_grad():
                ctx_chunk_reps = self.model(None, None, id_chunk, attn_chunk)[1]
            all_ctx_reps.append(ctx_chunk_reps)
        all_ctx_reps = torch.cat(all_ctx_reps)
            
        all_q_reps = all_q_reps.float().detach().requires_grad_()
        all_ctx_reps = all_ctx_reps.float().detach().requires_grad_()
        loss, num_correct = self.criterion.calc(all_q_reps, all_ctx_reps)
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        loss.backward()
            
        q_grads = all_q_reps.grad.split(self.args.q_chunk_size)
        ctx_grads = all_ctx_reps.grad.split(self.args.ctx_chunk_size)
            
        for id_chunk, attn_chunk, grad, rnd in zip(q_id_chunks, q_attn_mask_chunks, q_grads, q_rnds):
            with rnd:
                q_chunk_reps = self.model(id_chunk, attn_chunk, None, None)[0]
                surrogate = torch.dot(q_chunk_reps.flatten().float(), grad.flatten())
               #surrogate = surrogate * (trainer.distributed_factor / 8.)
            surrogate.backward()
                
        for id_chunk, attn_chunk, grad, rnd in zip(ctx_id_chunks, ctx_attn_mask_chunks, ctx_grads, ctx_rnds):
            with rnd:
                ctx_chunk_reps = self.model(None, None, id_chunk, attn_chunk)[1]
                surrogate = torch.dot(ctx_chunk_reps.flatten().float(), grad.flatten())
               #surrogate = surrogate * (trainer.distributed_factor / 8.)
            surrogate.backward()

       #q_vectors, ctx_vectors = self.model(q_input_ids, q_attn_mask, ctx_input_ids, ctx_attn_mask)
       #loss, num_correct = self.criterion.calc(q_vectors, ctx_vectors)
       #loss.backward()

        self.optimizer.step()
        self.scheduler.step()

        return loss.item(), num_correct
    
    def validate(self):
        self.model.eval()

        epoch_start = time.time()
        total_loss = 0.0
        total_correct = 0
        accuracy = 0

        with torch.no_grad():
            tqdm_batch_iterator = tqdm(self.val_loader)
            for i, batch in enumerate(tqdm_batch_iterator):
                if self.args.no_hard != 0:
                    q_input_ids, q_attn_mask, p_input_ids, p_attn_mask, n_input_ids, n_attn_mask = tuple(t.to(self.device) for t in batch)
                    ctx_len = n_input_ids.size()[-1]
                    n_input_ids = n_input_ids.view(-1,ctx_len)
                    n_attn_mask = n_attn_mask.view(-1,ctx_len)
                    ctx_input_ids = torch.cat((p_input_ids, n_input_ids), 0)
                    ctx_attn_mask = torch.cat((p_attn_mask, n_attn_mask), 0)
                else:
                    q_input_ids, q_attn_mask, ctx_input_ids, ctx_attn_mask = tuple(t.to(self.device) for t in batch)

                q_vectors, ctx_vectors = self.model(q_input_ids, q_attn_mask, ctx_input_ids, ctx_attn_mask)
                loss, num_correct = self.val_criterion.calc(q_vectors, ctx_vectors)
                total_loss += loss.item()
                total_correct += num_correct

            epoch_time = time.time() - epoch_start
            val_loss = total_loss/len(self.val_loader)
            accuracy = total_correct/len(self.val_loader.dataset)

        return epoch_time, val_loss, accuracy
